//+------------------------------------------------------------------+
//|                                                  LSTM_CNN_EA.mq5 |
//|                         LSTM-CNN XAUUSD Trading System          |
//|                    Dual-Model Expert Advisor with ONNX           |
//+------------------------------------------------------------------+
#property copyright "LSTM-CNN Trading System"
#property link      ""
#property version   "2.00"
#property strict

//+------------------------------------------------------------------+
//| Embedded Resources                                                |
//+------------------------------------------------------------------+
#resource "\\Models\\lstm_cnn_xauusd.onnx" as uchar PriceModelBuffer[]
#resource "\\Models\\profitability_classifier.onnx" as uchar ProfitModelBuffer[]
#resource "\\Models\\scalers.csv" as string ScalersCSV

//+------------------------------------------------------------------+
//| Include Modules                                                   |
//+------------------------------------------------------------------+
#include "Include/Feature_Engine.mqh"
#include "Include/ChartDisplay.mqh"
#include "Include/TradeManager.mqh"

//+------------------------------------------------------------------+
//| Input Parameters                                                  |
//+------------------------------------------------------------------+
input group "=== Model Settings ==="
input int    InpLookback = 30;                    // Lookback Window (bars)
input double InpMagnitudeThreshold = 0.5;         // Magnitude Threshold (ATR multiplier)
input double InpProfitThreshold = 0.40;           // Profitability Threshold (0.5-1.0)

input group "=== Trade Management ==="
input ulong  InpMagicNumber = 20250109;           // Magic Number
input double InpRiskPercent = 1.0;                // Risk Per Trade (%)
input int    InpMaxPositions = 3;                 // Max Concurrent Positions
input double InpMaxDailyLoss = 4.0;               // Max Daily Loss (%)
input double InpMaxDrawdown = 8.0;                // Max Total Drawdown (%)
input int    InpMaxBuysPerDay = 2;                // Max Buy Trades Per Day
input int    InpMaxSellsPerDay = 2;               // Max Sell Trades Per Day
input double InpATRMultiplierSL = 3.0;            // ATR Multiplier for Stop Loss
input double InpATRMultiplierTP = 6.0;            // ATR Multiplier for Take Profit
input int    InpATRPeriod = 14;                   // ATR Period

input group "=== Session Filter ==="
input bool   InpUseSessionFilter = true;          // Use Session Filter
input bool   InpTradeLondon = true;               // Trade London Session
input bool   InpTradeNewYork = true;              // Trade New York Session
input bool   InpTradeAsian = false;               // Trade Asian Session

input group "=== Trailing Stop ==="
input ENUM_TRAIL_TYPE InpTrailType = TRAIL_CHANDELIER; // Trailing Stop Type
input double InpTrailATRMultiplier = 2.5;         // Trailing ATR Multiplier

input group "=== Display Settings ==="
input bool   InpShowDisplay = true;               // Show Chart Display

//+------------------------------------------------------------------+
//| Feature indices matching training data (6 features after pruning) |
//+------------------------------------------------------------------+
#define NUM_FEATURES 6
// Training feature order: Volume(0), BB_Upper(1), RSI(2), MACD(3), MACD_Histogram(4), OBV(5)

//+------------------------------------------------------------------+
//| Global Variables                                                  |
//+------------------------------------------------------------------+
CFeatureEngine    g_feature_engine;
CChartDisplay     g_display;
CTradeManager     g_trade_manager;

long              g_price_model = INVALID_HANDLE;
long              g_profit_model = INVALID_HANDLE;
bool              g_ea_enabled = true;
datetime          g_last_bar_time = 0;
int               g_atr_handle = INVALID_HANDLE;
string            g_status = "Initializing...";

// Scalers from training
double            g_feature_min[NUM_FEATURES];
double            g_feature_max[NUM_FEATURES];
double            g_target_min = 0;
double            g_target_max = 0;

// Last prediction state
double            g_last_predicted_price = 0;
double            g_last_p_long = 0;
double            g_last_p_short = 0;
int               g_last_signal = 0;

//+------------------------------------------------------------------+
//| Parse scalers from embedded CSV                                   |
//+------------------------------------------------------------------+
bool LoadScalers()
{
   Print("Loading scalers from embedded CSV...");
   
   string lines[];
   int line_count = StringSplit(ScalersCSV, '\n', lines);
   
   if(line_count < 2)
   {
      Print("ERROR: Invalid scalers CSV format");
      return false;
   }
   
   for(int i = 1; i < line_count; i++)  // Skip header
   {
      string parts[];
      int part_count = StringSplit(lines[i], ',', parts);
      
      if(part_count < 3)
         continue;
      
      string name = parts[0];
      StringTrimLeft(name);
      StringTrimRight(name);
      double min_val = StringToDouble(parts[1]);
      double max_val = StringToDouble(parts[2]);
      
      // Match feature name to index (training order)
      if(name == "Volume")
      {
         g_feature_min[0] = min_val;
         g_feature_max[0] = max_val;
      }
      else if(name == "BB_Upper")
      {
         g_feature_min[1] = min_val;
         g_feature_max[1] = max_val;
      }
      else if(name == "RSI")
      {
         g_feature_min[2] = min_val;
         g_feature_max[2] = max_val;
      }
      else if(name == "MACD")
      {
         g_feature_min[3] = min_val;
         g_feature_max[3] = max_val;
      }
      else if(name == "MACD_Histogram")
      {
         g_feature_min[4] = min_val;
         g_feature_max[4] = max_val;
      }
      else if(name == "OBV")
      {
         g_feature_min[5] = min_val;
         g_feature_max[5] = max_val;
      }
      else if(name == "Target")
      {
         g_target_min = min_val;
         g_target_max = max_val;
      }
   }
   
   Print("Scalers loaded:");
   Print("  Target range: [", g_target_min, ", ", g_target_max, "]");
   for(int i = 0; i < NUM_FEATURES; i++)
      Print("  Feature ", i, ": [", g_feature_min[i], ", ", g_feature_max[i], "]");
   
   return true;
}


//+------------------------------------------------------------------+
//| Normalize a single feature value                                  |
//+------------------------------------------------------------------+
double NormalizeFeature(double value, int feature_idx)
{
   double range = g_feature_max[feature_idx] - g_feature_min[feature_idx];
   if(range == 0) return 0;
   return (value - g_feature_min[feature_idx]) / range;
}

//+------------------------------------------------------------------+
//| Denormalize predicted price                                       |
//+------------------------------------------------------------------+
double DenormalizePrice(double normalized_price)
{
   return normalized_price * (g_target_max - g_target_min) + g_target_min;
}

//+------------------------------------------------------------------+
//| Expert initialization function                                    |
//+------------------------------------------------------------------+
int OnInit()
{
   Print("=== LSTM-CNN Dual-Model EA v2.0 ===");
   
   //--- Load scalers from CSV
   if(!LoadScalers())
   {
      Print("CRITICAL: Failed to load scalers");
      g_ea_enabled = false;
      return INIT_FAILED;
   }
   
   //--- Load Price Predictor ONNX model
   Print("Loading Price Predictor model...");
   g_price_model = OnnxCreateFromBuffer(PriceModelBuffer, ONNX_DEFAULT);
   if(g_price_model == INVALID_HANDLE)
   {
      Print("CRITICAL: Failed to load Price Predictor model. Error: ", GetLastError());
      g_ea_enabled = false;
      return INIT_FAILED;
   }
   
   // Set input shape: [1, 30, 6]
   ulong price_input_shape[] = {1, (ulong)InpLookback, NUM_FEATURES};
   if(!OnnxSetInputShape(g_price_model, 0, price_input_shape))
   {
      Print("CRITICAL: Failed to set Price model input shape. Error: ", GetLastError());
      OnnxRelease(g_price_model);
      g_ea_enabled = false;
      return INIT_FAILED;
   }
   
   // Set output shape: [1, 1]
   ulong price_output_shape[] = {1, 1};
   if(!OnnxSetOutputShape(g_price_model, 0, price_output_shape))
   {
      Print("CRITICAL: Failed to set Price model output shape. Error: ", GetLastError());
      OnnxRelease(g_price_model);
      g_ea_enabled = false;
      return INIT_FAILED;
   }
   Print("Price Predictor loaded successfully");
   
   //--- Load Profitability Classifier ONNX model
   Print("Loading Profitability Classifier model...");
   g_profit_model = OnnxCreateFromBuffer(ProfitModelBuffer, ONNX_DEFAULT);
   if(g_profit_model == INVALID_HANDLE)
   {
      Print("CRITICAL: Failed to load Profitability model. Error: ", GetLastError());
      OnnxRelease(g_price_model);
      g_ea_enabled = false;
      return INIT_FAILED;
   }
   
   // Set input shape: [1, 30, 6]
   ulong profit_input_shape[] = {1, (ulong)InpLookback, NUM_FEATURES};
   if(!OnnxSetInputShape(g_profit_model, 0, profit_input_shape))
   {
      Print("CRITICAL: Failed to set Profit model input shape. Error: ", GetLastError());
      OnnxRelease(g_price_model);
      OnnxRelease(g_profit_model);
      g_ea_enabled = false;
      return INIT_FAILED;
   }
   
   // Set output shape: [1, 2] for [p_long, p_short]
   ulong profit_output_shape[] = {1, 2};
   if(!OnnxSetOutputShape(g_profit_model, 0, profit_output_shape))
   {
      Print("CRITICAL: Failed to set Profit model output shape. Error: ", GetLastError());
      OnnxRelease(g_price_model);
      OnnxRelease(g_profit_model);
      g_ea_enabled = false;
      return INIT_FAILED;
   }
   Print("Profitability Classifier loaded successfully");
   
   //--- Initialize Feature Engine
   if(!g_feature_engine.Init(_Symbol, PERIOD_CURRENT, InpLookback))
   {
      Print("CRITICAL: Failed to initialize Feature Engine");
      OnnxRelease(g_price_model);
      OnnxRelease(g_profit_model);
      g_ea_enabled = false;
      return INIT_FAILED;
   }
   Print("Feature Engine initialized");
   
   //--- Initialize ATR indicator
   g_atr_handle = iATR(_Symbol, PERIOD_CURRENT, InpATRPeriod);
   if(g_atr_handle == INVALID_HANDLE)
   {
      Print("WARNING: Failed to create ATR indicator");
   }
   
   //--- Initialize Trade Manager
   STradeConfig trade_config;
   trade_config.magicNumber = InpMagicNumber;
   trade_config.riskPercent = InpRiskPercent;
   trade_config.maxPositions = InpMaxPositions;
   trade_config.maxDailyLoss = InpMaxDailyLoss;
   trade_config.maxDrawdown = InpMaxDrawdown;
   trade_config.maxBuysPerDay = InpMaxBuysPerDay;
   trade_config.maxSellsPerDay = InpMaxSellsPerDay;
   trade_config.useSessionFilter = InpUseSessionFilter;
   trade_config.tradeLondon = InpTradeLondon;
   trade_config.tradeNewYork = InpTradeNewYork;
   trade_config.tradeAsian = InpTradeAsian;
   trade_config.trailType = InpTrailType;
   trade_config.atrMultiplier = InpTrailATRMultiplier;
   trade_config.atrPeriod = InpATRPeriod;
   
   if(!g_trade_manager.Init(_Symbol, PERIOD_CURRENT, trade_config))
   {
      Print("WARNING: Failed to initialize Trade Manager");
   }
   Print("Trade Manager initialized");
   
   //--- Initialize Chart Display
   if(InpShowDisplay)
   {
      SChartDisplayConfig display_config;
      display_config.eaName = "LSTM-CNN v2";
      display_config.modelName = "Dual-Model";
      display_config.confidenceThreshold = InpProfitThreshold;
      display_config.showProbabilities = true;
      display_config.showAccountInfo = true;
      display_config.showSessionInfo = InpUseSessionFilter;
      display_config.showPositionInfo = true;
      display_config.colorizeChart = true;
      display_config.maxPositions = InpMaxPositions;
      
      g_display.Init(display_config);
   }
   
   g_ea_enabled = true;
   g_status = "Ready";
   Print("=== Initialization Complete ===");
   
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                  |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   if(g_price_model != INVALID_HANDLE)
   {
      OnnxRelease(g_price_model);
      g_price_model = INVALID_HANDLE;
   }
   
   if(g_profit_model != INVALID_HANDLE)
   {
      OnnxRelease(g_profit_model);
      g_profit_model = INVALID_HANDLE;
   }
   
   if(g_atr_handle != INVALID_HANDLE)
   {
      IndicatorRelease(g_atr_handle);
      g_atr_handle = INVALID_HANDLE;
   }
   
   if(InpShowDisplay)
      g_display.ClearDisplay();
   
   Print("EA deinitialized. Reason: ", reason);
}

//+------------------------------------------------------------------+
//| Get ATR value                                                     |
//+------------------------------------------------------------------+
double GetATR(int shift = 0)
{
   if(g_atr_handle == INVALID_HANDLE) return 0;
   
   double atr[];
   ArraySetAsSeries(atr, true);
   if(CopyBuffer(g_atr_handle, 0, shift, 1, atr) != 1) return 0;
   return atr[0];
}


//+------------------------------------------------------------------+
//| Build input sequence for ONNX models                              |
//+------------------------------------------------------------------+
bool BuildInputSequence(float &input_data[])
{
   ArrayResize(input_data, InpLookback * NUM_FEATURES);
   
   // Get raw features for each bar in lookback window
   for(int bar = InpLookback; bar >= 1; bar--)
   {
      int seq_idx = InpLookback - bar;  // 0 to lookback-1
      int offset = seq_idx * NUM_FEATURES;
      
      // Get raw feature values
      double volume = (double)iVolume(_Symbol, PERIOD_CURRENT, bar);
      double bb_upper = 0, bb_middle = 0, bb_lower = 0;
      double rsi = 0, macd = 0, macd_signal = 0, macd_hist = 0, obv = 0;
      
      // Get BB Upper
      int bb_handle = iBands(_Symbol, PERIOD_CURRENT, 20, 0, 2.0, PRICE_CLOSE);
      if(bb_handle != INVALID_HANDLE)
      {
         double bb_buf[];
         if(CopyBuffer(bb_handle, 1, bar, 1, bb_buf) == 1)
            bb_upper = bb_buf[0];
         IndicatorRelease(bb_handle);
      }
      
      // Get RSI
      int rsi_handle = iRSI(_Symbol, PERIOD_CURRENT, 14, PRICE_CLOSE);
      if(rsi_handle != INVALID_HANDLE)
      {
         double rsi_buf[];
         if(CopyBuffer(rsi_handle, 0, bar, 1, rsi_buf) == 1)
            rsi = rsi_buf[0];
         IndicatorRelease(rsi_handle);
      }
      
      // Get MACD
      int macd_handle = iMACD(_Symbol, PERIOD_CURRENT, 12, 26, 9, PRICE_CLOSE);
      if(macd_handle != INVALID_HANDLE)
      {
         double macd_buf[], signal_buf[];
         if(CopyBuffer(macd_handle, 0, bar, 1, macd_buf) == 1)
            macd = macd_buf[0];
         if(CopyBuffer(macd_handle, 1, bar, 1, signal_buf) == 1)
            macd_signal = signal_buf[0];
         macd_hist = macd - macd_signal;
         IndicatorRelease(macd_handle);
      }
      
      // Get OBV
      int obv_handle = iOBV(_Symbol, PERIOD_CURRENT, VOLUME_TICK);
      if(obv_handle != INVALID_HANDLE)
      {
         double obv_buf[];
         if(CopyBuffer(obv_handle, 0, bar, 1, obv_buf) == 1)
            obv = obv_buf[0];
         IndicatorRelease(obv_handle);
      }
      
      // Normalize and store using training feature order (0-5)
      // Training order: Volume(0), BB_Upper(1), RSI(2), MACD(3), MACD_Histogram(4), OBV(5)
      input_data[offset + 0] = (float)NormalizeFeature(volume, 0);
      input_data[offset + 1] = (float)NormalizeFeature(bb_upper, 1);
      input_data[offset + 2] = (float)NormalizeFeature(rsi, 2);
      input_data[offset + 3] = (float)NormalizeFeature(macd, 3);
      input_data[offset + 4] = (float)NormalizeFeature(macd_hist, 4);
      input_data[offset + 5] = (float)NormalizeFeature(obv, 5);
   }
   
   return true;
}

//+------------------------------------------------------------------+
//| Run dual-model inference                                          |
//+------------------------------------------------------------------+
bool RunDualInference(double &predicted_price, double &p_long, double &p_short)
{
   predicted_price = 0;
   p_long = 0;
   p_short = 0;
   
   // Build input sequence
   float input_data[];
   if(!BuildInputSequence(input_data))
   {
      Print("ERROR: Failed to build input sequence");
      return false;
   }
   
   // Run Price Predictor
   float price_output[];
   ArrayResize(price_output, 1);
   
   if(!OnnxRun(g_price_model, ONNX_NO_CONVERSION, input_data, price_output))
   {
      Print("ERROR: Price model inference failed. Error: ", GetLastError());
      return false;
   }
   
   // Denormalize predicted price
   predicted_price = DenormalizePrice((double)price_output[0]);
   
   // Run Profitability Classifier
   float profit_output[];
   ArrayResize(profit_output, 2);
   
   if(!OnnxRun(g_profit_model, ONNX_NO_CONVERSION, input_data, profit_output))
   {
      Print("ERROR: Profit model inference failed. Error: ", GetLastError());
      return false;
   }
   
   p_long = (double)profit_output[0];
   p_short = (double)profit_output[1];
   
   return true;
}

//+------------------------------------------------------------------+
//| Get trading signal from dual-model output                         |
//| Returns: 0=Hold, 1=Buy, 2=Sell                                    |
//+------------------------------------------------------------------+
int GetTradingSignal(double predicted_price, double current_price, double atr,
                     double p_long, double p_short)
{
   // Calculate direction and magnitude
   double price_diff = predicted_price - current_price;
   int direction = (price_diff > 0) ? 1 : ((price_diff < 0) ? -1 : 0);
   double magnitude_in_atr = MathAbs(price_diff) / atr;
   
   Print("Signal Analysis:");
   Print("  Predicted: ", DoubleToString(predicted_price, _Digits));
   Print("  Current: ", DoubleToString(current_price, _Digits));
   Print("  Direction: ", (direction > 0 ? "UP" : (direction < 0 ? "DOWN" : "FLAT")));
   Print("  Magnitude: ", DoubleToString(magnitude_in_atr, 2), " ATR");
   Print("  P(Long wins): ", DoubleToString(p_long * 100, 1), "%");
   Print("  P(Short wins): ", DoubleToString(p_short * 100, 1), "%");
   
   // Check magnitude threshold
   if(magnitude_in_atr < InpMagnitudeThreshold)
   {
      Print("  -> HOLD: Magnitude below threshold (", InpMagnitudeThreshold, " ATR)");
      return 0;
   }
   
   // Check profitability threshold based on direction
   if(direction > 0)  // Predicted UP
   {
      if(p_long >= InpProfitThreshold)
      {
         Print("  -> BUY: Direction UP, P(long)=", DoubleToString(p_long * 100, 1), "% >= ", InpProfitThreshold * 100, "%");
         return 1;
      }
      else
      {
         Print("  -> HOLD: P(long)=", DoubleToString(p_long * 100, 1), "% < threshold");
         return 0;
      }
   }
   else if(direction < 0)  // Predicted DOWN
   {
      if(p_short >= InpProfitThreshold)
      {
         Print("  -> SELL: Direction DOWN, P(short)=", DoubleToString(p_short * 100, 1), "% >= ", InpProfitThreshold * 100, "%");
         return 2;
      }
      else
      {
         Print("  -> HOLD: P(short)=", DoubleToString(p_short * 100, 1), "% < threshold");
         return 0;
      }
   }
   
   return 0;
}

//+------------------------------------------------------------------+
//| Execute trading logic                                             |
//+------------------------------------------------------------------+
void ExecuteTrade(int signal)
{
   if(!g_ea_enabled || signal == 0)
      return;
   
   // Check risk limits
   g_trade_manager.CheckDailyReset();
   if(!g_trade_manager.CheckRiskLimits())
   {
      g_status = "Risk Limit";
      return;
   }
   
   // Check session
   if(!g_trade_manager.IsWithinTradingSession())
   {
      g_status = "Outside Session";
      return;
   }
   
   // Get ATR for SL/TP
   double atr = GetATR(0);
   if(atr <= 0)
   {
      g_status = "ATR Error";
      return;
   }
   
   double sl_distance = atr * InpATRMultiplierSL;
   double tp_distance = atr * InpATRMultiplierTP;
   
   if(signal == 1)  // Buy
   {
      double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      double sl = ask - sl_distance;
      double tp = ask + tp_distance;
      
      if(g_trade_manager.OpenBuy(sl, tp, "LSTM-CNN Buy"))
      {
         g_status = "Buy Opened";
         Print("BUY opened at ", ask, " SL=", sl, " TP=", tp);
      }
      else
         g_status = "Buy Failed";
   }
   else if(signal == 2)  // Sell
   {
      double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      double sl = bid + sl_distance;
      double tp = bid - tp_distance;
      
      if(g_trade_manager.OpenSell(sl, tp, "LSTM-CNN Sell"))
      {
         g_status = "Sell Opened";
         Print("SELL opened at ", bid, " SL=", sl, " TP=", tp);
      }
      else
         g_status = "Sell Failed";
   }
}

//+------------------------------------------------------------------+
//| Expert tick function                                              |
//+------------------------------------------------------------------+
void OnTick()
{
   if(!g_ea_enabled)
   {
      if(InpShowDisplay)
         g_display.UpdateSimpleDisplay(0, 0.0, "EA Disabled");
      return;
   }
   
   // Trail existing positions
   g_trade_manager.TrailStopLoss();
   
   // Only run on new bar
   datetime current_bar = iTime(_Symbol, PERIOD_CURRENT, 0);
   if(current_bar == g_last_bar_time)
   {
      // Update display with cached values
      if(InpShowDisplay)
      {
         g_display.UpdateDisplay(
            g_last_signal,
            g_last_p_long,
            g_last_p_short,
            GetATR(0),
            g_status,
            g_trade_manager.CountPositions(),
            g_trade_manager.GetCurrentSessionName(),
            g_trade_manager.IsWithinTradingSession()
         );
      }
      return;
   }
   
   g_last_bar_time = current_bar;
   Print("=== New Bar: ", TimeToString(current_bar), " ===");
   
   // Run dual-model inference
   double predicted_price, p_long, p_short;
   if(!RunDualInference(predicted_price, p_long, p_short))
   {
      g_status = "Inference Error";
      return;
   }
   
   // Get current price and ATR
   double current_price = iClose(_Symbol, PERIOD_CURRENT, 1);
   double atr = GetATR(1);
   
   if(atr <= 0)
   {
      g_status = "ATR Error";
      return;
   }
   
   // Get trading signal
   int signal = GetTradingSignal(predicted_price, current_price, atr, p_long, p_short);
   
   // Cache values
   g_last_predicted_price = predicted_price;
   g_last_p_long = p_long;
   g_last_p_short = p_short;
   g_last_signal = signal;
   
   // Execute trade
   if(signal != 0)
   {
      ExecuteTrade(signal);
   }
   else
   {
      g_status = "Holding";
   }
   
   // Update display
   if(InpShowDisplay)
   {
      g_display.UpdateDisplay(
         signal,
         p_long,
         p_short,
         atr,
         g_status,
         g_trade_manager.CountPositions(),
         g_trade_manager.GetCurrentSessionName(),
         g_trade_manager.IsWithinTradingSession()
      );
   }
}
//+------------------------------------------------------------------+
