# TSFDS: Time Series Forecast Data Set
#' Splits data into one training set and one or more test data sets.
#' @param data: Table of examples (one column per variable and one row per example).
#' @param col_time: Timestamp column index.
#' @param col_x_required: Required input column indexes.
#' @param col_x_optional: Optional input column indexes.
#' @param col_y: Output column indexes.
#' @param splitter: Row that splits training from test set (last training set row).
#' @param horizon: The length of forecast horizon (the number of time steps in the horizon).
#' @param lookback: The number of steps to look back in time to prepare past y values as forecast input.
#' @param uncertainty: The number of rows to shift back and forth while matching optional inputs with outputs.
#' @param encoding: One of available time encoding types:
#' 1) "polar": Polar coordinate system.
#' 2) "onehot": One-Hot encoding.
#' @param normalization: One of available normalization types:
#' 1) "AM": Abs-Max normalization
#' 2) "MM": Min-Max normalization
#' 3) "AVG": Divide-by-average normalization
#' 4) "Z": Z-normalization
#' @return A list with two members:
#' 1) train: A list with training data:
#' 1.1) x: Table with normalized training inputs.
#' 1.2) y: Table with normalized training outputs.
#' 1.3) norm: Normalization parameters (applied to both training and test data).
#' 2) test: A list with test data split by horizon, where each member contains:
#' 2.1) x: Table with normalized test inputs.
#' 2.2) y: Table with normalized test outputs.
TSFDS = function(data,
                 col_time,
                 col_x_required,
                 col_x_optional,
                 col_y,
                 splitter,
                 horizon,
                 lookback,
                 uncertainty = 1,
                 encoding = 'onehot',
                 normalization = 'MM') {
    PrepareSet = function() {
        Get = function(cols) return(Subset(data, cols = cols))
        Time = function() return(EncodeTime(DateTime(data[,col_time]),type=encoding))
        RequiredX = function() return(Get(col_x_required))
        OptionalX = function() {
            p = Get(col_x_optional)
            slider = -uncertainty:uncertainty
            candidates = as.data.frame(Slide(p, slider, FALSE))
            colnames(candidates) = Combine(colnames(p), slider)
            return(candidates)
        }

        y = Get(col_y)
        x.base = Join(Time(), RequiredX())
        x.opt = SelectFeatures(OptionalX(), y, TrainRows())
        x = Join(x.base, x.opt)
        return(list(x = x, y = y))
    }
    NormParams = function(x, y) {
        nx = NormParam(x, normalization, 'col')
        ny = NormParam(y, normalization, 'col')
        return(list(x = nx, y = ny))
    }
    TrainRows = function() return(1:splitter)
    TrainSubset = function(x, y) {
        x = Subset(x, rows=TrainRows())
        y = Subset(y, rows=TrainRows())
        norm = NormParams(x, y)
        x = Norm(x, norm$x)
        y = Norm(y, norm$y)
        return(list(x = x, y = y, norm = norm))
    }
    TestRows = function(splitter) return((splitter - lookback + 1):(splitter + horizon))
    TestSubsets = function(x, y, norm) {
        TestSubset = function(rows) {
            x = Norm(Subset(x, rows=rows), norm$x)
            y = Norm(Subset(y, rows=rows), norm$y)
            return(list(x = x, y = y))
        }

        test = list()
        for(i in 1:floor((Count(y)-splitter-uncertainty)/horizon)) {
            test[[i]] = TestSubset(TestRows(splitter+(i-1)*horizon))
        }

        return(test)
    }

    set = PrepareSet()
    train = TrainSubset(set$x, set$y)
    test = TestSubsets(set$x, set$y, train$norm)
    return(list(train = train, test = test))
}

#' Tensor: Creates n-dimensional dataset.
#' Creates a dataset with inputs and outputs for training or testing forecast model.
#' NOTE: The number of input and output examples must be the same (use NA to populate missing values).
#' @param x: Table of training input examples (one column per variable and one row per example).
#' @param y: Table of training output examples (one column per variable and one row per example).
#' @param horizon: The length of forecast horizon (the number of time steps in the horizon).
#' @param lookback: The number of steps to look back in time to prepare past y values as forecast input.
#' @param type: One of available tensor types:
#' 1) "MIMO": Tensor for Multiple-Input Multiple-Output forecast.
#' 2) "rec": Tensor for recursive (iterative) forecast.
#' @return A list, containing 3D arrays with examples, steps and variables:
#' 1) input: Input dataset (past y values combined with future x values).
#' 2) output: Output dataset (future y values).
Tensor = function(x, y, horizon, lookback, type = "MIMO") {
    x = as.data.frame(x)
    y = as.data.frame(y)
    count = min(Count(x), Count(y))
    future = (lookback + 1):count
    backward = (1-lookback):0
    if (type == "rec") {
        past = 1:(count - 1)
        forward = 0
    } else {
        past = 1:(count - horizon)
        forward = 0:(horizon-1)
    }

    input.past = Slide(y[past,], backward)
    input.future = Slide(x[future,], forward)
    input = list(input.past, input.future)
    output = Slide(y[future,], forward)
    return(list(input = input, output = output))
}

#' TrainSVR: Trains Support Vector Regression (SVR) model.
#' Creates recursive nu-based SVR model with Radial Basis Function (RBF) kernel.
#' Uses 10-Fold Cross Validation with Mean Square Error (MSE) to optimze the model.
#' NOTE: The number of input and output examples must be the same (use NA to populate missing values).
#' @param x: Table of training inputs (one column per variable and one row per example).
#' @param y: Table of training outputs (one column for only one variable and one row per example).
#' @param horizon (0, number of examples): The length of forecast horizon (the number of time steps in the horizon).
#' @param lookback (0, number of examples): The number of steps to look back in time to prepare past y values as forecast input.
#' @param nu (0, 1]: Parameter nu controls the number of support vectors and training errors for nu-SVR as follows:
#' 1) small nu leads to great sensitivity to margins around hyperplanes and may cause overfitting,
#' 2) large nu leads to small sensitivity to margins around hyperplanes and may cause underfitting.
#' In general, nu-SVR and epsilon-SVR are equivalent but nu parameter is more intuitive that epsilon.
#' For example, if nu = 0.1, then atmost 10% of the training examples will be considered as outliers
#  and atleast 10% of training examples will act as support vectors (points on the decision boundary).
#' @param gamma [0, infinity]: Parameter for RBF kernel controls the shape of the separating hyperplanes as follows:
#' 1) Small gamma leads to a Gaussian function with a large variance (margins) and may cause underfitting,
#' 2) Large gamma leads to a Gaussian function with a small variance (margins) and may cause overfitting.
#' @param cost (0, infinity]: Parameter C for SVR penalizes errors and controls the complexity of SVR as follows:
#' 1) Small cost leads to large margins around hyperplanes and may cause underfitting,
#' 2) Large cost leads to small margins around hyperplanes and may cause overfitting.
#' @param tolerance (0, 1): Convergence tolerance controls the search for an optimal solution (hyperplanes) as follows:
#' 1) Small tolerance increases difficulty in finding an optimal solution and may cause overfitting,
#' 2) Large tolerance decreases difficulty in finding an optimal solution and may cause underfitting.
#' @param verbose: Shows whether or not to log training details.
#' @return SVR model.
TrainSVR = function(x, y, horizon, lookback = 2*horizon, nu = 0.01, gamma = 0.001, cost = 1, tolerance = 0.01, verbose = FALSE) {
    LimitParams = function() {
        nu <<- Limit(nu, 0.001, 1)
        gamma <<- Limit(gamma, min = 0)
        cost <<- Limit(cost, min = 0.001)
        tolerance <<- Limit(tolerance, 0.001, 0.999)
    }
    PrintParams = function() {
        names = c("targets",
                  "predictors",
                  "horizon",
                  "lookback",
                  "nu",
                  "gamma",
                  "cost",
                  "tolerance")
        values = c(paste(ncol(y)),
                   ncol(x),
                   horizon,
                   lookback,
                   nu,
                   gamma,
                   cost,
                   tolerance)
        names(values) = names
        print(values)
    }

    LimitParams()
    if (verbose) {
        Log(c("SVR training started..."))
        stopwatch = StopwatchStartNew()
        PrintParams()
    }

    train = Tensor(x, y, horizon, lookback, "rec")
    input = as.matrix(as.data.frame(train$input))
    output = as.matrix(as.data.frame(train$output))
    model = svm(x = input, y = output, type = "nu-regression", nu = nu, gamma = gamma, cost = cost, tolerance = tolerance, cross = 10)

    if (verbose) Log(c("SVR training finished (duration = ", StopwatchElapsedSeconds(stopwatch), " second(s))."))
    return(model)
}

#' TestSVR: Tests Support Vector Regression (SVR) model.
#' NOTE: The number of input and output examples must be the same (use NA to populate missing values).
#' @param model: Trained SVR forecast model.
#' @param x: Table of future values for test input (one column per variable and one row per example).
#' @param y: Table of past values for test output (one column per variable and one row per example).
#' @param horizon (0, number of examples): The length of forecast horizon (the number of time steps in the horizon).
#' @param lookback (0, number of examples): The number of steps to look back in time to prepare past y values as forecast input.
#' @param verbose: Shows whether or not to log forecast errors and plot forecast results.
#' @return A list containing:
#' 1) forecast: Time series forecast.
#' 2) errors: Forecast errors.
TestSVR = function(model, x, y, horizon, lookback, verbose = FALSE) {
    test = Tensor(x, y, horizon, lookback, "rec")
    input = as.matrix(as.data.frame(test$input))
    actual = as.matrix(as.data.frame(test$output))
    input.column = ncol(input) - ncol(as.data.frame(x))

    forecast = c()
    for (i in 1:horizon) {
        if (i > 1) input[i, input.column] = forecast[i - 1]
        output = predict(model, matrix(input[i,], 1, ncol(input)))
        forecast = c(forecast, output)
    }

    forecast = as.data.frame(forecast)
    colnames(forecast) = colnames(y)
    rownames(forecast) = c()
    errors = EvaluateForecast(actual, forecast, verbose = verbose)
    return(list(forecast = forecast, errors = errors))
}

#' OptimizeSVR: Optimized Support Vector Regression (SVR) training.
#' Metaheuristic optimization of forecast model hyperparameters.
#' @param x: Table of training input examples (one column per variable and one row per example).
#' @param y: Table of training output examples (one column per variable and one row per example).
#' @param horizon (0, number of examples): The length of forecast horizon (the number of time steps in the horizon).
#' @param lookback (0, number of examples): The number of steps to look back in time to prepare past y values as forecast input.
#' @param optimizer: One of available metaOpt algorithms.
#' @param population (1, infinity]: Optimizer population size.
#' @param generations (1, infinity]: Maximum number of optimization iterations.
#' @param other: Other optimizer parameters.
#' @param folder: Folder for csv export with best parameters and errors.
#' @param name: Name of parameters and errors (csv file name prefix).
#' @param verbose: Shows whether or not to log forecast errors and plot forecast results.
#' @return Search result.
OptimizeSVR = function(x,
                       y,
                       horizon,
                       lookback = 2*horizon,
                       optimizer = "GOA",
                       population = 10,
                       generations = 20,
                       other = list(),
                       folder = NULL,
                       name = NULL,
                       verbose = FALSE) {
    PrepareSet = function(x, y, horizon, lookback) {
        x = as.data.frame(x)
        y = as.data.frame(y)
        train = 1:(Count(y) - horizon)
        valid = (Count(y) - horizon - lookback + 1):Count(y)
        return(list(x.train = as.data.frame(x[train,]),
                    y.train = as.data.frame(y[train,]),
                    x.valid = as.data.frame(x[valid,]),
                    y.valid = as.data.frame(y[valid,])))
    }
    Evaluate = function(args) {
        names(args) = c("nu", "gamma", "cost", "tolerance")
        nu = args["nu"]
        gamma = args["gamma"]
        cost = args["cost"]
        tolerance = args["tolerance"]

        model = TrainSVR(set$x.train, set$y.train, horizon, lookback, nu, gamma, cost, tolerance, verbose)
        output = model %>% TestSVR(set$x.valid, set$y.valid, horizon, lookback, verbose)

        error = output$errors["RMSE[%]"]
        if (is.null(best.error) || error < best.error) {
            best.args <<- args
            best.model <<- model
            best.error <<- error
            if (!is.null(folder)) {
                ExportCSV(as.data.frame(best.args), folder, paste(name, "-args", sep = ""), rows = TRUE, separator = "=")
                ExportCSV(as.data.frame(best.error), folder, paste(name, "-error", sep = ""), rows = TRUE, separator = "=")
            }
        }

        return(error)
    }

    if (verbose) {
        Log(c(optimizer, "-SVR optimization started (", generations, "x", population, " iterations)..."))
        stopwatch = StopwatchStartNew()
    }

    optimal = NULL
    best.args = NULL
    best.model = NULL
    best.error = NULL
    set = PrepareSet(x, y, horizon, lookback)

    # nu, gamma, cost, tolerance:
    min = c(0, 0, 0.1, 0)
    max = c(1, 1, 1, 0.1)
    optimal = Optimize(Evaluate, min, max, optimizer, population, generations, other)

    if (verbose) {
        Log(c("SVR optimization finished (duration = ", StopwatchElapsedSeconds(stopwatch), " second(s))."))
        print(optimal)
        print(best.args)
        print(best.error)
    }

    return(best.model)
}

#' TrainRNN: Trains Recurrent Neural Network (RNN).
#' Creates Multiple-Input Multiple-Output (MIMO) RNN:
#' Bidirectional Sequence to Sequence (S2S) Long Short-Term Memory (LSTM) with attention mechanism.
#' Adaptive moment estimation (Adam) optimization is used to minimize loss function: Mean Squared Error (MSE).
#' Coefficient of Variation of the Root MSE (CV-RMSE) [%] is used for evaluating model and early stopping.
#' NOTE: The number of input and output examples must be the same (use NA to populate missing values).
#' @param x: Table of training input examples (one column per variable and one row per example).
#' @param y: Table of training output examples (one column per variable and one row per example).
#' @param horizon (0, number of examples): The length of forecast horizon (the number of time steps in the horizon).
#' @param lookback (0, number of examples): The number of steps to look back in time to prepare past y values as forecast input.
#' @param context (0, 1): The size of the S2S context relative to the number of lookbacks.
#' @param fnns [0, infinity]: The number of fully-connected Feedforward Neural Network (FNN) layers inside S2S context autoencoder.
#' @param fnnm (0, 1): The size of the FNN layer in the middle of the S2S context autoencoder relative to the edges of autoencoder.
#' @param cnns [0, infinity]: The number of Convolution Neural Network (CNN) layers prepended to S2S model.
#' @param cnnf [1, infinity]: The number of convolutional filters in each CNN layer.
#' @param cnnk [2, number of lookbacks): The size of convolutional filters in each CNN layer.
#' @param cnnp [2, number of lookbacks): The size of pools in average pooling layers within CNN layers.
#' @param fnndrop [0, 1): Fraction of the units to drop after each FNN layer (FNN dropout).
#' @param cnndrop [0, 1): Fraction of the units to drop after each CNN layer (CNN dropout).
#' @param iterations (1, infinity]: Maximum number of training iterations.
#' @param batch (1, infinity]: Batch size for each iteration.
#' @param verbose: Shows whether or not to log and plot training details.
#' @return RNN model.
TrainRNN = function(x,
                    y,
                    horizon,
                    lookback = 2*horizon,
                    context = 0.7,
                    fnns = round(horizon / 3),
                    fnnm = 0.3,
                    cnns = 1,
                    cnnf = 512,
                    cnnk = 3,
                    cnnp = 3,
                    fnndrop = 0.5,
                    cnndrop = 0.5,
                    iterations = 50,
                    batch = 1024,
                    verbose = FALSE) {
    LimitParams = function() {
        x <<- as.data.frame(x)
        y <<- as.data.frame(y)
        horizon <<- Limit(horizon, 1, Count(y) - 1)
        lookback <<- Limit(lookback, 1, Count(y) - 1)
        context <<- Limit(context, 0.1, 0.9)
        fnns <<- Limit(fnns, min = 0)
        fnnm <<- Limit(fnnm, 0.1, 0.9)
        cnnf <<- Limit(cnnf, min = 1)
        cnnk <<- Limit(cnnk, 2, lookback - 1)
        cnnp <<- Limit(cnnk, 2, lookback - 1)
        cnns <<- Limit(cnns, 0, floor(log(lookback/(cnnk - 1), base = cnnp)))
        fnndrop <<- Limit(fnndrop, 0, 0.9)
        cnndrop <<- Limit(cnndrop, 0, 0.9)
    }
    PrintParams = function() {
        names = c("targets",
                  "predictors",
                  "horizon",
                  "lookback",
                  "context",
                  "FNN layers",
                  "FNN middle",
                  "CNN layers",
                  "CNN filters",
                  "CNN kernel",
                  "CNN pool",
                  "FNN dropout",
                  "CNN dropout")
        values = c(paste(ncol(y)),
                   ncol(x),
                   horizon,
                   lookback,
                   context,
                   fnns,
                   fnnm,
                   cnns,
                   cnnf,
                   cnnk,
                   cnnp,
                   fnndrop,
                   cnndrop)
        names(values) = names
        print(values)
    }

    AbsInt = function(rel, max) return(Limit(round(rel * max), 1, max - 1))

    FNN = function(input, units, name, id) {
        name.feed = paste(name, "_FNN", id, sep = "")
        name.drop = paste(name, "_FNN", id, "_dropout", sep = "")
        out = input %>%
              layer_dense(units = units,
                          activation = "tanh",
                          name = name.feed) %>%
              layer_dropout(fnndrop, name = name.drop)
        return(out)
    }
    CNN = function(input, name, id) {
        name.conv = paste(name, "_CNN", id, "_convolution", sep = "")
        name.pool = paste(name, "_CNN", id, "_pooling", sep = "")
        name.drop = paste(name, "_CNN", id, "_dropout", sep = "")
        out = input %>%
              layer_conv_1d(filters = cnnf,
                            kernel_size = cnnk,
                            activation = "relu",
                            padding = "same",
                            name = name.conv) %>%
              layer_average_pooling_1d(pool_size = cnnp, name = name.pool) %>%
              layer_dropout(cnndrop, name = name.drop)
        return(out)
    }
    LSTM = function(units, states = FALSE, sequences = FALSE, name = NULL) {
        return(layer_lstm(units = units,
                          return_state = states,
                          return_sequences = sequences,
                          name = name,
                          # GPU requirement:
                          activation = "tanh",
                          recurrent_activation = "sigmoid",
                          recurrent_dropout = 0,
                          use_bias = TRUE,
                          unroll = FALSE))
    }

    AutoEncoder = function(input, edges, middle, depth, name) {
        FNNUnits = function(min, max, depth) {
            if (depth <= 0) return(c())
            if (depth == 1) return(c(max))
            if (depth == 2) return(c(max, max))

            units = c(max)
            half = (depth - 1) / 2
            step = floor((max - min) / floor(half))
            for (i in 1:floor(half)) {
                units = c(units, Limit(units[i] - step, min, max))
            }

            units = c(units, rev(units[1:ceiling(half)]))
            return(units)
        }

        units = FNNUnits(AbsInt(middle, edges), edges, depth)
        if (Count(units) == 0) return(input)
        for (i in 1:Count(units)) input = input %>% FNN(units[i], name, i)
        return(input)
    }
    DeepCNN = function(input, depth, name) {
        if (depth == 0) return(input)
        for (i in 1:depth) input = input %>% CNN(name, i)
        return(input)
    }
    S2S = function(past, future) {
        Encode = function(encoder.input, units) {
            encoder = LSTM(units, states = TRUE)
            encoded = encoder.input %>% bidirectional(encoder, name = "LSTM_encoder")
            return(encoded)
        }
        Decode = function(encoded, units, decoder.input) {
            EncodedOutput = function() return(encoded[1])
            EncodedStates = function() {
                # Extracting and autoencoding 2 hidden and 2 cell states from bidirectional encoder:
                h = layer_concatenate(encoded[c(2, 4)], name = "S2S_hState")
                c = layer_concatenate(encoded[c(3, 5)], name = "S2S_cState")
                h = h %>% AutoEncoder(units, fnnm, fnns, name = "S2S_hState")
                c = c %>% AutoEncoder(units, fnnm, fnns, name = "S2S_cState")
                return(list(h, c))
            }

            decoder = LSTM(units, sequences = TRUE, name = "LSTM_decoder")
            decoded = decoder.input %>% decoder(initial_state = EncodedStates())
            attention = layer_attention(list(decoded, EncodedOutput()), name = "S2S_attention")
            output = layer_concatenate(list(decoded, attention), name = "S2S_output")
            return(output)
        }

        past = past %>% DeepCNN(cnns, "Past")
        units = AbsInt(context, lookback)
        return(Encode(past, units) %>% Decode(2 * units, future))
    }

    Create = function() {
        Input = function(variables, steps, name) return(layer_input(shape = list(steps, variables), name = name))
        InputPast = function() return(Input(ncol(y), lookback, "input_past"))
        InputFuture = function() return(Input(ncol(x), horizon, "input_future"))
        Output = function(model) return(layer_dense(model, units = ncol(y), name = "output"))

        past = InputPast()
        future = InputFuture()
        output = S2S(past, future) %>% Output()
        model = keras_model(inputs = list(past, future), outputs = output)
        return(model)
    }
    Compile = function(model) {
        CVRMSE = function(y_true, y_pred) {
            RMSE = k_sqrt(metric_mean_squared_error(y_true, y_pred))
            return(100 * RMSE / k_mean(y_true))
        }

        model %>% compile(loss = 'mse', optimizer = "adam", metrics = CVRMSE)
        if (verbose) model %>% summary()
        return(model)
    }

    LimitParams()
    if (verbose) {
        Log(c("RNN training started..."))
        stopwatch = StopwatchStartNew()
        PrintParams()
    }

    model = Create() %>% Compile() %>% UpdateRNN(x, y, horizon, lookback, iterations, batch, verbose)

    if (verbose) Log(c("RNN training finished (duration = ", StopwatchElapsedSeconds(stopwatch), " second(s))."))
    return(model)
}

#' UpdateRNN: Updates Recurrent Neural Network (RNN).
#' Uses 70:30 split ratio for separating training from validation data.
#' NOTE: The number of input and output examples must be the same (use NA to populate missing values).
#' @param model: Trained Multiple-Input Multiple-Output (MIMO) RNN forecast model.
#' @param x: Table of training input examples (one column per variable and one row per example).
#' @param y: Table of training output examples (one column per variable and one row per example).
#' @param horizon (0, number of examples): The length of forecast horizon (the number of time steps in the horizon).
#' @param lookback (0, number of examples): The number of steps to look back in time to prepare past y values as forecast input.
#' @param iterations (1, infinity]: Maximum number of training iterations.
#' @param batch (1, infinity]: Batch size for each iteration.
#' @param verbose: Shows whether or not to log and plot training details.
#' @return Updated RNN model.
UpdateRNN = function(model, x, y, horizon, lookback, iterations = 50, batch = 1024, verbose = FALSE) {
    train = Tensor(x, y, horizon, lookback, "MIMO")
    limit = callback_early_stopping(monitor = "python_function", mode = "min", patience = 10)
    split = ifelse(dim(train$output)[1] < 10, 0, 0.3)
    model %>% fit(train$input, train$output,
                  epochs = max(1, iterations), batch_size = max(1, batch), callbacks = list(limit),
                  validation_split = split, shuffle = FALSE, verbose = as.numeric(verbose))
    return(model)
}

#' TestRNN: Tests Recurrent Neural Network (RNN).
#' NOTE: The number of input and output examples must be the same (use NA to populate missing values).
#' @param model: Trained Multiple-Input Multiple-Output (MIMO) RNN forecast model.
#' @param x: Table of future values for test input (one column per variable and one row per example).
#' @param y: Table of past values for test output (one column per variable and one row per example).
#' @param horizon (0, number of examples): The length of forecast horizon (the number of time steps in the horizon).
#' @param lookback (0, number of examples): The number of steps to look back in time to prepare past y values as forecast input.
#' @param verbose: Shows whether or not to log forecast errors and plot forecast results.
#' @return A list containing:
#' 1) forecast: Time series forecast.
#' 2) errors: Forecast errors.
TestRNN = function(model, x, y, horizon, lookback, verbose = FALSE) {
    GetOutput = function(tensor) {
        variables = ncol(as.data.frame(y))
        data = matrix(tensor, ncol = variables, nrow = horizon)
        output = as.data.frame(data)
        colnames(output) = colnames(y)
        rownames(output) = c()
        return(output)
    }

    test = Tensor(x, y, horizon, lookback, "MIMO")
    result = model %>% predict(test$input)
    actual = GetOutput(test$output)
    forecast = GetOutput(result)
    errors = EvaluateForecast(actual, forecast, verbose = verbose)
    return(list(forecast = forecast, errors = errors))
}

#' SaveRNN: Saves Recurrent Neural Network (RNN) to '.rnn' file.
#' @param model: Trained RNN forecast model.
#' @param folder: Destination folder.
#' @param file: Destination file name.
SaveRNN = function(model, folder, file) model %>% save_model_hdf5(GetPath(folder, file, 'rnn'))

#' LoadRNN: Loads Recurrent Neural Network (RNN) from '.rnn' file.
#' @param folder: Source folder.
#' @param file: Source file name.
#' @return Trained RNN forecast model.
LoadRNN = function(folder, file) return(load_model_hdf5(GetPath(folder, file, 'rnn')))

#' OptimizeRNN: Optimized Recurrent Neural Network (RNN) training.
#' Metaheuristic optimization of forecast model hyperparameters.
#' @param x: Table of training input examples (one column per variable and one row per example).
#' @param y: Table of training output examples (one column per variable and one row per example).
#' @param horizon (0, number of examples): The length of forecast horizon (the number of time steps in the horizon).
#' @param lookback (0, number of examples): The number of steps to look back in time to prepare past y values as forecast input.
#' @param optimizer: One of available metaOpt algorithms.
#' @param population (1, infinity]: Optimizer population size.
#' @param generations (1, infinity]: Maximum number of optimization iterations.
#' @param iterations (1, infinity]: Maximum number of training iterations.
#' @param batch (1, infinity]: Batch size for each iteration.
#' @param other: Other optimizer parameters.
#' @param folder: Folder for csv export with best parameters and errors.
#' @param name: Name of parameters and errors (csv file name prefix).
#' @param verbose: Shows whether or not to log forecast errors and plot forecast results.
#' @return Search result.
OptimizeRNN = function(x,
                       y,
                       horizon,
                       lookback = 2*horizon,
                       optimizer = "GOA",
                       population = 10,
                       generations = 20,
                       iterations = 20,
                       batch = 1024,
                       other = list(),
                       folder = NULL,
                       name = NULL,
                       verbose = FALSE) {
    Prepare = function(x, y, horizon, lookback) {
        x = as.data.frame(x)
        y = as.data.frame(y)
        train = 1:(Count(y) - horizon)
        valid = (Count(y) - horizon - lookback + 1):Count(y)
        return(list(x.train = as.data.frame(x[train,]),
                    y.train = as.data.frame(y[train,]),
                    x.valid = as.data.frame(x[valid,]),
                    y.valid = as.data.frame(y[valid,])))
    }
    Evaluate = function(args) {
        context = args[1]
        fnns = round(args[2])
        fnnm = args[3]
        cnns = round(args[4])
        cnnf = round(args[5])
        cnnk = round(args[6])
        cnnp = round(args[7])
        fnndrop = args[8]
        cnndrop = args[9]

        args = c(context, fnns, fnnm, cnns, cnnf, cnnk, cnnp, fnndrop, cnndrop)
        names(args) = c("context", "fnns", "fnnm", "cnns", "cnnf", "cnnk", "cnnp", "fnndrop", "cnndrop")

        model = TrainRNN(set$x.train, set$y.train, horizon, lookback,
                         context, fnns, fnnm, cnns, cnnf, cnnk, cnnp, fnndrop, cnndrop,
                         iterations, batch, verbose)
        output = model %>% TestRNN(set$x.valid, set$y.valid, horizon, lookback, verbose)

        error = output$errors["RMSE[%]"]
        if (is.null(best.error) || error < best.error) {
            best.args <<- args
            best.model <<- model
            best.error <<- error
            if (!is.null(folder)) {
                ExportCSV(as.data.frame(best.args), folder, paste(name, "-args", sep = ""), rows = TRUE, separator = "=")
                ExportCSV(as.data.frame(best.error), folder, paste(name, "-error", sep = ""), rows = TRUE, separator = "=")
            }
        } else {
            rm(model)
            k_clear_session()
        }

        return(error)
    }

    if (verbose) {
        Log(c(optimizer, "-RNN optimization started (", generations, "x", population, " iterations)..."))
        stopwatch = StopwatchStartNew()
    }

    best.args = NULL
    best.model = NULL
    best.error = NULL
    set = Prepare(x, y, horizon, lookback)

    # context, fnns, fnnm, cnns, cnnf, cnnk cnnp, fnndrop cnndrop:
    min = c(0.4, 0, 0.2, 0, 128, 2, 2, 0.2, 0.2)
    max = c(0.8, 9, 0.8, 5, 768, 5, 5, 0.6, 0.6)
    optimal = Optimize(Evaluate, min, max, optimizer, population, generations, other)

    if (verbose) {
        Log(c("RNN optimization finished (duration = ", StopwatchElapsedSeconds(stopwatch), " second(s))."))
        print(optimal)
        print(best.args)
        print(best.error)
    }

    return(best.model)
}

#' EvaluateForecast: Calculates forecast errors.
#' Summarizes deterministic and/or probabilistic forecast errors.
#' @param actual: Actual values.
#' @param expected: Expected values (means).
#' @param stdev: Expected standard deviations (used only for probabilistic forecast).
#' @param n [0, infinity]: Sample size for each mean and standard deviation (used only for probabilistic forecast).
#' @param p [0, 1]: Probabilisty level for mean and standard deviation (used only for probabilistic forecast).
#' @param verbose: Shows whether or not to log forecast errors and plot forecast results.
#' @return Vector of forecast errors:
#' 1) "MAPE[%]": Mean Absolute Percentage Error for deterministic forecast.
#' 3) "RMSE[%]": Coefficient of Variation of the Root Mean Square Error for deterministic forecast.
#' 5) "CRPS": Continuous Ranked Probability Score for probabilistic forecast.
#' 3) "ACE[%]": Average Coverage Error for probabilistic forecast.
#' 4) "WS": Winkler Score for probabilistic forecast.
EvaluateForecast = function(actual, expected, stdev = c(), n = 0, p = 0.95, verbose = FALSE) {
    PrepareParams = function() {
        PI = function() {
            i = pi.m(M = expected, SD = stdev, n = n, prob.level = p)
            return(list(min = i$lower_prediction_interval, max = i$upper_prediction_interval))
        }

        actual <<- as.vector(unlist(actual))
        expected <<- as.vector(unlist(expected))
        stdev <<- as.vector(unlist(stdev))
        n <<- Limit(n, min = 0)
        p <<- Limit(p, 0, 1)
        i <<- PI()
    }

    Evaluate = function() {
        MAPE = function() return(100 * mean(abs(actual - expected) / actual))
        CVRMSE = function() {
            rmse = sqrt(mean((actual - expected) ^ 2))
            return(100 * rmse / mean(actual))
        }
        CRPS = function() return(crps(actual, Join(expected, stdev))$CRPS)
        ACE = function() {
            covered = as.numeric(i$min <= actual & actual <= i$max)
            return(100 * (p - mean(covered)))
        }
        WS = function() {
            Winkler = function() {
                alpha = 1 - p
                diff = (i$max - i$min)
                if (actual < i$min) return(diff + 2 * (i$min - actual) / alpha)
                if (actual > i$max) return(diff + 2 * (actual - i$max) / alpha)
                return(diff)
            }

            return(mean(Winkler()))
        }

        if (Count(actual) == 0 || Count(expected) == 0) return(c())
        deterministic = c("MAPE[%]" = MAPE(), "RMSE[%]" = CVRMSE())
        if (Count(stdev) == 0 || n == 0) return(deterministic)
        probabilistic = c("ACE[%]" = ACE(), "WS" = WS(), "CRPS" = CRPS())
        return(c(deterministic, probabilistic))
    }

    Plot = function(errors) {
        Summary = function(errors) {
            words = Join(names(errors), paste("=", round(errors, 3)))
            return(paste(as.vector(t(as.matrix(words))), collapse = " "))
        }
        MakePlot = function(title) {
            x.axis = 1:max(Count(actual), Count(expected))
            y.values = na.omit(c(actual, expected, i$min, i$max))
            plot(x = x.axis, ylim = c(min(y.values), max(y.values)), type = "n", main = title)
            return(x.axis)
        }
        AddLine = function(x, y, color) {
            if (Count(x) != Count(y)) return()
            lines(x, y, type = "l", col = color, lwd = 2)
        }
        AddLegend = function(items, colors) legend("topright", legend = items, col = colors, lty = 1, cex = 0.8)

        x.axis = MakePlot(Summary(errors))

        items = c()
        colors = c()
        if (Count(na.omit(actual)) > 0) {
            AddLine(x.axis, actual, "blue")
            items = c(items, "actual")
            colors = c(colors, "blue")
        }
        if (Count(na.omit(expected)) > 0) {
            AddLine(x.axis, expected, "red")
            items = c(items, "expected")
            colors = c(colors, "red")
        }
        if (Count(na.omit(stdev)) > 0 && n > 0) {
            AddLine(x.axis, i$min, "pink")
            AddLine(x.axis, i$max, "pink")
            items = c(items, paste(p * 100, "% PI"))
            colors = c(colors, "pink")
        }

        AddLegend(items, colors)
    }

    PrepareParams()
    errors = Evaluate()
    if (verbose && Count(errors) > 0) {
        print(errors)
        Plot(errors)
    }

    return(errors)
}
