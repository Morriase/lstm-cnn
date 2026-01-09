//+------------------------------------------------------------------+
//|                                                  LSTM_CNN_EA.mq5 |
//|                         LSTM-CNN XAUUSD Trading System          |
//|                    Expert Advisor with ONNX Inference            |
//+------------------------------------------------------------------+
#property copyright "LSTM-CNN Trading System"
#property link      ""
#property version   "1.00"
#property strict

//+------------------------------------------------------------------+
//| Embedded ONNX Model Resource                                      |
//| Requirements: 8.1, 11.1, 11.2                                     |
//+------------------------------------------------------------------+
#resource "\\Models\\lstm_cnn_xauusd.onnx" as uchar OnnxModelBuffer[]

//+------------------------------------------------------------------+
//| Include Modules                                                   |
//+------------------------------------------------------------------+
#include "Include/Feature_Engine.mqh"
#include "Include/Correlation_Pruner.mqh"
#include "Include/ChartDisplay.mqh"
#include "Include/TradeManager.mqh"  // Temporarily disabled for compilation

//+------------------------------------------------------------------+
//| Input Parameters - Model Settings                                 |
//| Requirements: 11.1, 11.2                                          |
//+------------------------------------------------------------------+
input group "=== Model Settings ==="
input int    InpLookback = 30;                    // Lookback Window (bars)
input double InpConfidenceThreshold = 0.55;       // Confidence Threshold for Trading

input group "=== Feature Engineering ==="
input int    InpSMA_Period1 = 10;                 // SMA Period 1
input int    InpSMA_Period2 = 50;                 // SMA Period 2
input int    InpEMA_Period1 = 10;                 // EMA Period 1
input int    InpEMA_Period2 = 50;                 // EMA Period 2
input int    InpBB_Period = 20;                   // Bollinger Bands Period
input double InpBB_Deviation = 2.0;               // Bollinger Bands Deviation
input int    InpRSI_Period = 14;                  // RSI Period
input int    InpMACD_Fast = 12;                   // MACD Fast Period
input int    InpMACD_Slow = 26;                   // MACD Slow Period
input int    InpMACD_Signal = 9;                  // MACD Signal Period

input group "=== Normalization Bounds (from training) ==="
input double InpMinOpen = 1200.0;                 // Min Open Price
input double InpMaxOpen = 2100.0;                 // Max Open Price
input double InpMinHigh = 1200.0;                 // Min High Price
input double InpMaxHigh = 2100.0;                 // Max High Price
input double InpMinLow = 1200.0;                  // Min Low Price
input double InpMaxLow = 2100.0;                  // Max Low Price
input double InpMinClose = 1200.0;                // Min Close Price
input double InpMaxClose = 2100.0;                // Max Close Price
input double InpMinVolume = 0.0;                  // Min Volume
input double InpMaxVolume = 100000.0;             // Max Volume
input double InpMinSMA10 = 1200.0;                // Min SMA 10
input double InpMaxSMA10 = 2100.0;                // Max SMA 10
input double InpMinSMA50 = 1200.0;                // Min SMA 50
input double InpMaxSMA50 = 2100.0;                // Max SMA 50
input double InpMinEMA10 = 1200.0;                // Min EMA 10
input double InpMaxEMA10 = 2100.0;                // Max EMA 10
input double InpMinEMA50 = 1200.0;                // Min EMA 50
input double InpMaxEMA50 = 2100.0;                // Max EMA 50
input double InpMinBBUpper = 1200.0;              // Min BB Upper
input double InpMaxBBUpper = 2200.0;              // Max BB Upper
input double InpMinBBMiddle = 1200.0;             // Min BB Middle
input double InpMaxBBMiddle = 2100.0;             // Max BB Middle
input double InpMinBBLower = 1100.0;              // Min BB Lower
input double InpMaxBBLower = 2100.0;              // Max BB Lower
input double InpMinRSI = 0.0;                     // Min RSI
input double InpMaxRSI = 100.0;                   // Max RSI
input double InpMinMACD = -50.0;                  // Min MACD
input double InpMaxMACD = 50.0;                   // Max MACD
input double InpMinMACDSignal = -50.0;            // Min MACD Signal
input double InpMaxMACDSignal = 50.0;             // Max MACD Signal
input double InpMinMACDHist = -30.0;              // Min MACD Histogram
input double InpMaxMACDHist = 30.0;               // Max MACD Histogram
input double InpMinOBV = -1000000000.0;           // Min OBV
input double InpMaxOBV = 1000000000.0;            // Max OBV

input group "=== Correlation Pruning ==="
input double InpCorrelationThreshold = 0.85;      // Correlation Threshold
input string InpRetainedIndices = "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16"; // Retained Feature Indices (comma-separated)

input group "=== Trade Management ==="
input ulong  InpMagicNumber = 20250109;           // Magic Number
input double InpRiskPercent = 1.0;                // Risk Per Trade (%)
input int    InpMaxPositions = 3;                 // Max Concurrent Positions
input double InpMaxDailyLoss = 4.0;               // Max Daily Loss (%)
input double InpMaxDrawdown = 8.0;                // Max Total Drawdown (%)
input int    InpMaxBuysPerDay = 2;                // Max Buy Trades Per Day
input int    InpMaxSellsPerDay = 2;               // Max Sell Trades Per Day
input double InpATRMultiplierSL = 2.0;            // ATR Multiplier for Stop Loss
input double InpATRMultiplierTP = 3.0;            // ATR Multiplier for Take Profit
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
input bool   InpColorizeChart = true;             // Colorize Chart by Sentiment

//+------------------------------------------------------------------+
//| Global Variables                                                  |
//+------------------------------------------------------------------+
CFeatureEngine    g_feature_engine;               // Feature engineering module
CCorrelationPruner g_pruner;                      // Correlation pruning module
CChartDisplay     g_display;                      // Chart display module
CTradeManager     g_trade_manager;                // Trade management module

long              g_onnx_handle = INVALID_HANDLE; // ONNX model handle
bool              g_ea_enabled = true;            // EA enabled flag
datetime          g_last_bar_time = 0;            // Last processed bar time
int               g_atr_handle = INVALID_HANDLE;  // ATR indicator handle
int               g_retained_indices[];           // Retained feature indices after pruning
int               g_num_retained_features = 0;    // Number of retained features

// Prediction state
int               g_last_prediction = 0;          // 0=Hold, 1=Buy, 2=Sell
double            g_last_confidence = 0.0;        // Prediction confidence
string            g_status = "Initializing...";   // Current status message

//+------------------------------------------------------------------+
//| Parse retained indices from input string                          |
//+------------------------------------------------------------------+
bool ParseRetainedIndices(string indices_str, int &indices[])
{
   string parts[];
   int count = StringSplit(indices_str, ',', parts);
   
   if(count <= 0)
   {
      Print("ParseRetainedIndices: No indices found in string");
      return false;
   }
   
   ArrayResize(indices, count);
   
   for(int i = 0; i < count; i++)
   {
      StringTrimLeft(parts[i]);
      StringTrimRight(parts[i]);
      indices[i] = (int)StringToInteger(parts[i]);
      
      // Validate index is within feature count range
      if(indices[i] < 0 || indices[i] >= FEAT_COUNT)
      {
         Print("ParseRetainedIndices: Invalid index ", indices[i], " at position ", i);
         return false;
      }
   }
   
   return true;
}

//+------------------------------------------------------------------+
//| Set normalization bounds from input parameters                    |
//+------------------------------------------------------------------+
bool SetNormalizationBoundsFromInputs()
{
   double min_vals[FEAT_COUNT];
   double max_vals[FEAT_COUNT];
   
   // Set bounds for each feature from input parameters
   min_vals[FEAT_OPEN] = InpMinOpen;           max_vals[FEAT_OPEN] = InpMaxOpen;
   min_vals[FEAT_HIGH] = InpMinHigh;           max_vals[FEAT_HIGH] = InpMaxHigh;
   min_vals[FEAT_LOW] = InpMinLow;             max_vals[FEAT_LOW] = InpMaxLow;
   min_vals[FEAT_CLOSE] = InpMinClose;         max_vals[FEAT_CLOSE] = InpMaxClose;
   min_vals[FEAT_VOLUME] = InpMinVolume;       max_vals[FEAT_VOLUME] = InpMaxVolume;
   min_vals[FEAT_SMA_10] = InpMinSMA10;        max_vals[FEAT_SMA_10] = InpMaxSMA10;
   min_vals[FEAT_SMA_50] = InpMinSMA50;        max_vals[FEAT_SMA_50] = InpMaxSMA50;
   min_vals[FEAT_EMA_10] = InpMinEMA10;        max_vals[FEAT_EMA_10] = InpMaxEMA10;
   min_vals[FEAT_EMA_50] = InpMinEMA50;        max_vals[FEAT_EMA_50] = InpMaxEMA50;
   min_vals[FEAT_BB_UPPER] = InpMinBBUpper;    max_vals[FEAT_BB_UPPER] = InpMaxBBUpper;
   min_vals[FEAT_BB_MIDDLE] = InpMinBBMiddle;  max_vals[FEAT_BB_MIDDLE] = InpMaxBBMiddle;
   min_vals[FEAT_BB_LOWER] = InpMinBBLower;    max_vals[FEAT_BB_LOWER] = InpMaxBBLower;
   min_vals[FEAT_RSI] = InpMinRSI;             max_vals[FEAT_RSI] = InpMaxRSI;
   min_vals[FEAT_MACD] = InpMinMACD;           max_vals[FEAT_MACD] = InpMaxMACD;
   min_vals[FEAT_MACD_SIGNAL] = InpMinMACDSignal; max_vals[FEAT_MACD_SIGNAL] = InpMaxMACDSignal;
   min_vals[FEAT_MACD_HISTOGRAM] = InpMinMACDHist; max_vals[FEAT_MACD_HISTOGRAM] = InpMaxMACDHist;
   min_vals[FEAT_OBV] = InpMinOBV;             max_vals[FEAT_OBV] = InpMaxOBV;
   
   return g_feature_engine.SetNormalizationBounds(min_vals, max_vals);
}

//+------------------------------------------------------------------+
//| Expert initialization function                                    |
//| Requirements: 8.1, 8.2, 8.3                                       |
//+------------------------------------------------------------------+
int OnInit()
{
   Print("=== LSTM-CNN EA Initialization ===");
   
   //--- Step 1: Load ONNX model from embedded resource buffer
   Print("Loading ONNX model from embedded resource...");
   
   g_onnx_handle = OnnxCreateFromBuffer(OnnxModelBuffer, ONNX_DEFAULT);
   
   if(g_onnx_handle == INVALID_HANDLE)
   {
      int error = GetLastError();
      Print("CRITICAL: Failed to load ONNX model from buffer. Error: ", error);
      g_ea_enabled = false;
      g_status = "ONNX Load Failed";
      return INIT_FAILED;
   }
   
   Print("ONNX model loaded successfully. Handle: ", g_onnx_handle);
   
   //--- Step 2: Parse retained feature indices
   if(!ParseRetainedIndices(InpRetainedIndices, g_retained_indices))
   {
      Print("CRITICAL: Failed to parse retained indices");
      OnnxRelease(g_onnx_handle);
      g_onnx_handle = INVALID_HANDLE;
      g_ea_enabled = false;
      g_status = "Invalid Indices";
      return INIT_FAILED;
   }
   
   g_num_retained_features = ArraySize(g_retained_indices);
   Print("Retained features: ", g_num_retained_features);
   
   //--- Step 3: Set ONNX input/output shapes
   // Input shape: [batch_size, lookback, num_features]
   ulong input_shape[] = {1, (ulong)InpLookback, (ulong)g_num_retained_features};
   
   if(!OnnxSetInputShape(g_onnx_handle, 0, input_shape))
   {
      int error = GetLastError();
      Print("CRITICAL: Failed to set ONNX input shape. Error: ", error);
      OnnxRelease(g_onnx_handle);
      g_onnx_handle = INVALID_HANDLE;
      g_ea_enabled = false;
      g_status = "Shape Error";
      return INIT_FAILED;
   }
   
   Print("ONNX input shape set: [1, ", InpLookback, ", ", g_num_retained_features, "]");
   
   // Output shape: [batch_size, 1] for price prediction
   ulong output_shape[] = {1, 1};
   
   if(!OnnxSetOutputShape(g_onnx_handle, 0, output_shape))
   {
      int error = GetLastError();
      Print("CRITICAL: Failed to set ONNX output shape. Error: ", error);
      OnnxRelease(g_onnx_handle);
      g_onnx_handle = INVALID_HANDLE;
      g_ea_enabled = false;
      g_status = "Shape Error";
      return INIT_FAILED;
   }
   
   Print("ONNX output shape set: [1, 1]");
   
   //--- Step 4: Initialize Feature Engine
   if(!g_feature_engine.Init(_Symbol, PERIOD_CURRENT, InpLookback))
   {
      Print("CRITICAL: Failed to initialize Feature Engine");
      OnnxRelease(g_onnx_handle);
      g_onnx_handle = INVALID_HANDLE;
      g_ea_enabled = false;
      g_status = "Feature Engine Error";
      return INIT_FAILED;
   }
   
   // Set feature engineering parameters to match training
   g_feature_engine.SetSMAPeriods(InpSMA_Period1, InpSMA_Period2);
   g_feature_engine.SetEMAPeriods(InpEMA_Period1, InpEMA_Period2);
   g_feature_engine.SetBBParams(InpBB_Period, InpBB_Deviation);
   g_feature_engine.SetRSIPeriod(InpRSI_Period);
   g_feature_engine.SetMACDParams(InpMACD_Fast, InpMACD_Slow, InpMACD_Signal);
   
   // Set normalization bounds from input parameters
   if(!SetNormalizationBoundsFromInputs())
   {
      Print("WARNING: Failed to set normalization bounds");
   }
   
   Print("Feature Engine initialized");
   
   //--- Step 5: Initialize Correlation Pruner
   if(!g_pruner.Init(InpCorrelationThreshold))
   {
      Print("CRITICAL: Failed to initialize Correlation Pruner");
      OnnxRelease(g_onnx_handle);
      g_onnx_handle = INVALID_HANDLE;
      g_ea_enabled = false;
      g_status = "Pruner Error";
      return INIT_FAILED;
   }
   
   // Set retained indices from input parameters
   if(!g_pruner.SetRetainedIndices(g_retained_indices))
   {
      Print("CRITICAL: Failed to set retained indices in pruner");
      OnnxRelease(g_onnx_handle);
      g_onnx_handle = INVALID_HANDLE;
      g_ea_enabled = false;
      g_status = "Pruner Error";
      return INIT_FAILED;
   }
   
   Print("Correlation Pruner initialized");
   
   //--- Step 6: Initialize ATR indicator for SL/TP calculation
   g_atr_handle = iATR(_Symbol, PERIOD_CURRENT, InpATRPeriod);
   if(g_atr_handle == INVALID_HANDLE)
   {
      Print("WARNING: Failed to create ATR indicator");
   }
   
   //--- Step 7: Initialize Trade Manager
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
   
   //--- Step 8: Initialize Chart Display
   if(InpShowDisplay)
   {
      SChartDisplayConfig display_config;
      display_config.eaName = "LSTM-CNN";
      display_config.modelName = "XAUUSD Predictor";
      display_config.confidenceThreshold = InpConfidenceThreshold;
      display_config.showProbabilities = true;
      display_config.showAccountInfo = true;
      display_config.showSessionInfo = InpUseSessionFilter;
      display_config.showPositionInfo = true;
      display_config.colorizeChart = InpColorizeChart;
      display_config.maxPositions = InpMaxPositions;
      
      g_display.Init(display_config);
      Print("Chart Display initialized");
   }
   
   g_ea_enabled = true;
   g_status = "Ready";
   g_last_bar_time = 0;
   
   Print("=== LSTM-CNN EA Initialization Complete ===");
   
   return INIT_SUCCEEDED;
}


//+------------------------------------------------------------------+
//| Expert deinitialization function                                  |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   Print("=== LSTM-CNN EA Deinitialization ===");
   
   //--- Release ONNX model handle
   if(g_onnx_handle != INVALID_HANDLE)
   {
      OnnxRelease(g_onnx_handle);
      g_onnx_handle = INVALID_HANDLE;
      Print("ONNX model released");
   }
   
   //--- Release ATR indicator handle
   if(g_atr_handle != INVALID_HANDLE)
   {
      IndicatorRelease(g_atr_handle);
      g_atr_handle = INVALID_HANDLE;
   }
   
   //--- Clear chart display
   if(InpShowDisplay)
   {
      g_display.ClearDisplay();
   }
   
   Print("Deinit reason: ", reason);
}

//+------------------------------------------------------------------+
//| Get ATR value for SL/TP calculation                               |
//+------------------------------------------------------------------+
double GetATRValue(int shift = 0)
{
   if(g_atr_handle == INVALID_HANDLE)
      return 0.0;
   
   double atr[];
   ArraySetAsSeries(atr, true);
   
   if(CopyBuffer(g_atr_handle, 0, shift, 1, atr) != 1)
      return 0.0;
   
   return atr[0];
}

//+------------------------------------------------------------------+
//| Run ONNX inference and get prediction                             |
//| Requirements: 8.4, 8.5                                            |
//+------------------------------------------------------------------+
bool RunInference(double &prediction)
{
   prediction = 0.0;
   
   //--- Step 1: Compute feature sequence using Feature_Engine
   double raw_sequence[];
   if(!g_feature_engine.ComputeSequence(1, raw_sequence))
   {
      Print("RunInference: Failed to compute feature sequence");
      return false;
   }
   
   //--- Step 2: Normalize the features
   // The sequence is [lookback * FEAT_COUNT] - normalize each feature
   int seq_size = ArraySize(raw_sequence);
   for(int i = 0; i < InpLookback; i++)
   {
      int offset = i * FEAT_COUNT;
      double bar_features[];
      ArrayResize(bar_features, FEAT_COUNT);
      
      // Extract features for this bar
      for(int j = 0; j < FEAT_COUNT; j++)
      {
         bar_features[j] = raw_sequence[offset + j];
      }
      
      // Normalize
      if(!g_feature_engine.NormalizeFeatures(bar_features))
      {
         Print("RunInference: Failed to normalize features at bar ", i);
         return false;
      }
      
      // Copy back
      for(int j = 0; j < FEAT_COUNT; j++)
      {
         raw_sequence[offset + j] = bar_features[j];
      }
   }
   
   //--- Step 3: Apply correlation pruning to get retained features only
   // Create pruned sequence with only retained features
   int pruned_size = InpLookback * g_num_retained_features;
   float input_data[];
   ArrayResize(input_data, pruned_size);
   
   for(int i = 0; i < InpLookback; i++)
   {
      int raw_offset = i * FEAT_COUNT;
      int pruned_offset = i * g_num_retained_features;
      
      // Extract only retained features
      for(int j = 0; j < g_num_retained_features; j++)
      {
         int feature_idx = g_retained_indices[j];
         input_data[pruned_offset + j] = (float)raw_sequence[raw_offset + feature_idx];
      }
   }
   
   //--- Step 4: Run ONNX inference
   float output_data[];
   ArrayResize(output_data, 1);
   
   if(!OnnxRun(g_onnx_handle, ONNX_NO_CONVERSION, input_data, output_data))
   {
      int error = GetLastError();
      Print("RunInference: ONNX inference failed. Error: ", error);
      return false;
   }
   
   prediction = (double)output_data[0];
   return true;
}

//+------------------------------------------------------------------+
//| Convert price prediction to trading signal                        |
//| Returns: 0=Hold, 1=Buy, 2=Sell                                    |
//+------------------------------------------------------------------+
int GetTradingSignal(double prediction, double current_price, double &confidence)
{
   // Calculate predicted price change percentage
   double price_change_pct = (prediction - current_price) / current_price * 100.0;
   
   // Use absolute percentage change as confidence
   confidence = MathAbs(price_change_pct);
   
   // Normalize confidence to 0-1 range (assuming max 2% move is high confidence)
   confidence = MathMin(confidence / 2.0, 1.0);
   
   // Determine signal based on predicted direction
   if(price_change_pct > 0.05)  // Predicted price increase > 0.05%
      return 1;  // Buy signal
   else if(price_change_pct < -0.05)  // Predicted price decrease > 0.05%
      return 2;  // Sell signal
   else
      return 0;  // Hold - no clear direction
}

//+------------------------------------------------------------------+
//| Execute trading logic based on prediction                         |
//+------------------------------------------------------------------+
void ExecuteTradingLogic(int signal, double confidence)
{
   //--- Check if trading is allowed
   if(!g_ea_enabled)
      return;
   
   //--- Check risk limits
   g_trade_manager.CheckDailyReset();
   if(!g_trade_manager.CheckRiskLimits())
   {
      g_status = "Risk Limit Hit";
      return;
   }
   
   //--- Check session filter
   if(!g_trade_manager.IsWithinTradingSession())
   {
      g_status = "Outside Session";
      return;
   }
   
   //--- Check confidence threshold
   if(confidence < InpConfidenceThreshold)
   {
      g_status = "Low Confidence";
      return;
   }
   
   //--- Get ATR for SL/TP calculation
   double atr = GetATRValue(0);
   if(atr <= 0)
   {
      g_status = "ATR Error";
      return;
   }
   
   double sl_distance = atr * InpATRMultiplierSL;
   double tp_distance = atr * InpATRMultiplierTP;
   
   //--- Execute trade based on signal
   if(signal == 1)  // Buy signal
   {
      double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      double sl = ask - sl_distance;
      double tp = ask + tp_distance;
      
      if(g_trade_manager.OpenBuy(sl, tp, "LSTM-CNN Buy"))
      {
         g_status = "Buy Opened";
         Print("Buy trade opened. Prediction confidence: ", DoubleToString(confidence * 100, 1), "%");
      }
      else
      {
         g_status = "Buy Failed";
      }
   }
   else if(signal == 2)  // Sell signal
   {
      double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      double sl = bid + sl_distance;
      double tp = bid - tp_distance;
      
      if(g_trade_manager.OpenSell(sl, tp, "LSTM-CNN Sell"))
      {
         g_status = "Sell Opened";
         Print("Sell trade opened. Prediction confidence: ", DoubleToString(confidence * 100, 1), "%");
      }
      else
      {
         g_status = "Sell Failed";
      }
   }
   else
   {
      g_status = "Holding";
   }
}

//+------------------------------------------------------------------+
//| Expert tick function                                              |
//| Requirements: 8.4, 8.5, 8.6, 8.7                                  |
//+------------------------------------------------------------------+
void OnTick()
{
   //--- Check if EA is enabled
   if(!g_ea_enabled)
   {
      if(InpShowDisplay)
         g_display.UpdateSimpleDisplay(0, 0.0, "EA Disabled");
      return;
   }
   
   //--- Trail existing positions
   g_trade_manager.TrailStopLoss();
   
   //--- Detect new bar - only run inference on new bar
   datetime current_bar_time = iTime(_Symbol, PERIOD_CURRENT, 0);
   if(current_bar_time == g_last_bar_time)
   {
      // Update display with last prediction (no new inference)
      if(InpShowDisplay)
      {
         double prob_buy = (g_last_prediction == 1) ? g_last_confidence : (1.0 - g_last_confidence) / 2.0;
         double prob_sell = (g_last_prediction == 2) ? g_last_confidence : (1.0 - g_last_confidence) / 2.0;
         
         g_display.UpdateDisplay(
            g_last_prediction,
            prob_buy,
            prob_sell,
            GetATRValue(0),
            g_status,
            g_trade_manager.CountPositions(),
            g_trade_manager.GetCurrentSessionName(),
            g_trade_manager.IsWithinTradingSession()
         );
      }
      return;
   }
   
   g_last_bar_time = current_bar_time;
   Print("New bar detected: ", TimeToString(current_bar_time));
   
   //--- Run ONNX inference
   double prediction = 0.0;
   if(!RunInference(prediction))
   {
      g_status = "Inference Error";
      Print("ERROR: ONNX inference failed");
      
      if(InpShowDisplay)
         g_display.UpdateSimpleDisplay(0, 0.0, g_status);
      return;
   }
   
   //--- Get current price for comparison
   double current_price = iClose(_Symbol, PERIOD_CURRENT, 1);
   
   //--- Convert prediction to trading signal
   double confidence = 0.0;
   int signal = GetTradingSignal(prediction, current_price, confidence);
   
   //--- Store prediction state
   g_last_prediction = signal;
   g_last_confidence = confidence;
   
   Print("Prediction: ", DoubleToString(prediction, _Digits), 
         ", Current: ", DoubleToString(current_price, _Digits),
         ", Signal: ", (signal == 1 ? "BUY" : (signal == 2 ? "SELL" : "HOLD")),
         ", Confidence: ", DoubleToString(confidence * 100, 1), "%");
   
   //--- Execute trading logic
   ExecuteTradingLogic(signal, confidence);
   
   //--- Update chart display
   if(InpShowDisplay)
   {
      double prob_buy = (signal == 1) ? confidence : (1.0 - confidence) / 2.0;
      double prob_sell = (signal == 2) ? confidence : (1.0 - confidence) / 2.0;
      
      g_display.UpdateDisplay(
         signal,
         prob_buy,
         prob_sell,
         GetATRValue(0),
         g_status,
         g_trade_manager.CountPositions(),
         g_trade_manager.GetCurrentSessionName(),
         g_trade_manager.IsWithinTradingSession()
      );
   }
}

//+------------------------------------------------------------------+
