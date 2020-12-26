# TSFDS: Time Series Forecast Data Set
#' Splits data into one training set and one or more test data sets.
#' @param data: Table of examples (one column per variable and one row per example).
#' @param col_time: Timestamp column index.
#' @param cols_x_required: Required input column indexes.
#' @param cols_x_optional: Optional input column indexes.
#' @param col_y: Output column index.
#' @param horizon: The length of forecast horizon (the number of time steps in the horizon).
#' @param lookback: The number of steps to look back in time to prepare past y values as forecast input.
#' @param splitter: Row that splits training from test set (last training set row).
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
                 cols_x_required,
                 cols_x_optional,
                 col_y,
                 horizon,
                 lookback = 2*horizon,
                 splitter = round(Count(data)*0.7),
                 uncertainty = 1,
                 encoding = 'onehot',
                 normalization = 'MM') {
    PrepareSet = function() {
        Get = function(cols) return(Subset(data, cols = cols))
        Time = function() return(EncodeTime(DateTime(data[,col_time]),type=encoding))
        RequiredX = function() return(Get(cols_x_required))
        OptionalX = function() {
            p = Get(cols_x_optional)
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
        set = list(x = x, y = y)
        return(list(set = set, norm = norm))
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
    return(list(train = train$set, test = test, norm = train$norm))
}

# TSF: Time Series Forecast
#' @param data: Table of examples (one column per variable and one row per example).
#' @param col_time: Timestamp column index.
#' @param cols_x_required: Required input column indexes.
#' @param cols_x_optional: Optional input column indexes.
#' @param cols_y: Output column indexes.
#' @param horizon: The length of forecast horizon (the number of time steps in the horizon).
#' @param lookback: The number of steps to look back in time to prepare past y values as forecast input.
#' @param splitter: Row that splits training from test set (last training set row).
#' @param type: One of available types of forecast models:
#' 1) "RNN": Recurrent Neural Network.
#' 2) "SVR": Support Vector Regression.
#' @param uncertainty: The number of rows to shift back and forth while matching optional inputs with outputs.
#' @param encoding: One of available time encoding types:
#' 1) "polar": Polar coordinate system.
#' 2) "onehot": One-Hot encoding.
#' @param normalization: One of available normalization types:
#' 1) "AM": Abs-Max normalization
#' 2) "MM": Min-Max normalization
#' 3) "AVG": Divide-by-average normalization
#' 4) "Z": Z-normalization
#' @param optimizer: One of available metaheuristic hyperparameter optimizers from metaOpt package.
#' The 70:30 split ratio is used for separating training from validation data during optimization.
#' Set the optimizer to NULL if the optimization is not required.
#' @param population (1, infinity]: Optimizer population size.
#' @param generations (1, infinity]: Maximum number of optimization iterations.
#' @param optparam: Other optimizer parameters.
#' @param verbose: Shows whether or not to log details.
#' @param folder: Export folder.
#' @param ...: Non-default training hyperparameters to use when optimization is not required.
#' These are SVR- or RNN-specific parameters, used by TrainSVR or TrainRNN function, respectively.
#' @return Table of forecasts.
TSF = function(data,
               col_time,
               cols_x_required,
               cols_x_optional,
               cols_y,
               horizon,
               lookback = 2*horizon,
               splitter = round(Count(data)*0.7),
               type = 'RNN',
               uncertainty = 1,
               encoding = 'onehot',
               normalization = 'MM',
               optimizer = "GOA",
               population = 10,
               generations = 10,
               optparam = list(),
               verbose = FALSE,
               folder = NULL,
               ...) {
    TryExport = function(table, file) {
        if (!is.null(folder)) table %>% ExportCSV(folder, name, verbose = verbose)
    }
    Forecast = function(col_y) {
        GetName = function() return(ifelse(Count(colnames(data)) == 0, col_y, colnames(data)[col_y]))
        TestIndexSplitter = function() return(round(splitter / horizon * 0.3))
        TrainTo = function() return(splitter - TestIndexSplitter() * horizon)
        ValidateFrom = function() return(TrainTo() + 1)
        TestFrom = function() return(splitter + 1)

        Dataset = function() {
            set = TSFDS(data,
                        col_time,
                        cols_x_required,
                        cols_x_optional,
                        col_y,
                        horizon,
                        lookback,
                        TrainTo(),
                        uncertainty,
                        encoding,
                        normalization)
            return(set)
        }

        CanSave = function() return(type == 'RNN' && !is.null(folder))
        Save = function(model) SaveRNN(model, folder, GetName())
        Load = function() return(LoadRNN(folder, GetName()))

        TrySave = function(model) {
            if (CanSave()) model %>% Save()
        }
        TryLoad = function() {
            if (!CanSave()) return(NULL)
            tryCatch({ return(Load()) }, error = function(e){ return(NULL) })
        }

        Train = function(set) {
            TrainModel = function() {
                if (type == 'SVR') {
                    return(TrainSVR(set$train$x, set$train$y, horizon, lookback, ...))
                } else {
                    return(TrainRNN(set$train$x, set$train$y, horizon, lookback, ..., verbose = verbose))
                }
            }

            model = TrainModel()
            model %>% TrySave()
            return(model)
        }
        Test = function(model, set, i) {
            NormForecast = function(x, y) {
                if (type == 'SVR') {
                    return(model %>% TestSVR(x, y, horizon, lookback))
                } else {
                    return(model %>% TestRNN(x, y, horizon, lookback))
                }
            }

            test = set$test[[i]]
            nres = NormForecast(test$x, test$y)
            return(Norm(nres, set$norm$y, FALSE))
        }
        Update = function(model, set, i, args = c()) {
            if (type == 'SVR') {
                x = set$train$x
                y = set$train$y
                for (t in 1:i) {
                    test = set$test[[t]]
                    rows = (Count(test$x)-horizon+1):Count(test$x)
                    x = Union(x, test$x[rows,])
                    y = Union(y, test$y[rows,])
                }

                if (Count(args) == 0) {
                    return(TrainSVR(x, y, horizon, lookback, ...))
                } else {
                    return(TrainSVR(x, y, horizon, lookback,
                                    args$nu,
                                    args$gamma,
                                    args$cost,
                                    args$tolerance))
                }
            } else {
                test = set$test[[i]]
                model %>% UpdateRNN(test$x, test$y, horizon, lookback, verbose = verbose)
                return(model)
            }
        }
        Errors = function(from, forecast) {
            to = from + Count(forecast) - 1
            actual = data[from:to, col_y]
            errors = EvaluateForecast(actual, forecast, verbose = verbose)
            return(errors)
        }
        OptTrain = function(set) {
            best = list(args = c(), model = NULL, error = NA)

            Evaluate = function(model, args) {
                forecast = data.frame()
                for (i in 1:TestIndexSplitter()) {
                    forecast = Union(forecast, model %>% Test(set, i))
                    model = model %>% Update(set, i)
                }

                error = Errors(ValidateFrom(), forecast)["RMSE[%]"]
                if (is.na(best$error) || error < best$error) {
                    best$args = args
                    best$model = model
                    best$error = error

                    TrySave(best$model)
                    TryExport(best$args, paste(GetName(), "args", sep = "-"))
                    TryExport(best$error, paste(GetName(), "error", sep = "-"))
                } else if (type == 'RNN') {
                    rm(model)
                    k_clear_session()
                }

                return(error)
            }
            EvaluateRNN = function(args) {
                names(args) = c("context", "fnns", "fnnm", "cnns", "cnnf", "cnnk", "cnnp", "fnndrop", "cnndrop")
                args = as.list(args)
                model = TrainRNN(x        = set$train$x,
                                 y        = set$train$y,
                                 horizon  = horizon,
                                 lookback = lookback,
                                 context  = args$context,
                                 fnns     = args$fnns,
                                 fnnm     = args$fnnm,
                                 cnns     = args$cnns,
                                 cnnf     = args$cnnf,
                                 cnnk     = args$cnnk,
                                 cnnp     = args$cnnp,
                                 fnndrop  = args$fnndrop,
                                 cnndrop  = args$cnndrop,
                                 verbose  = verbose)
                return(model %>% Evaluate(args))
            }
            OptimizeRNN = function(){
                # context, fnns, fnnm, cnns, cnnf, cnnk cnnp, fnndrop cnndrop:
                min = c(0.6, 0, 0.2, 0, 256, 2, 2, 0.4, 0.4)
                max = c(0.8, 9, 0.5, 5, 768, 4, 4, 0.6, 0.6)
                Optimize(EvaluateRNN, min, max, optimizer, population, generations, optparam)
            }
            EvaluateSVR = function(args) {
                names(args) = c("nu", "gamma", "cost", "tolerance")
                args = as.list(args)
                model = TrainSVR(set$x.train,
                                 set$y.train,
                                 horizon,
                                 lookback,
                                 args$nu,
                                 args$gamma,
                                 args$cost,
                                 args$tolerance)
                return(model %>% Evaluate(args))
            }
            OptimizeSVR = function() {
                # nu, gamma, cost, tolerance:
                min = c(0, 0, 0.1, 0)
                max = c(1, 1, 1, 0.1)
                Optimize(EvaluateSVR, min, max, optimizer, population, generations, optparam)
            }

            if (type == 'SVR') {
                OptimizeSVR()
            } else {
                OptimizeRNN()
            }

            return(best)
        }
        TryOptTrain = function(set) {
            best = list(args = c(), model = TryLoad(), error = NA)
            if (!is.null(best$model)) {
                Log(c(type, " model loaded."))
                return(best)
            }
            if (is.null(optimizer)) {
                watch = Log(c(type, " training started..."))

                best$model = set %>% Train()

                Log(c(type, " training finished (duration = ", Elapsed(watch), " sec)."))
                return(best)
            }

            watch = Log(c(optimizer, "-", type, " optimization started (", generations, "x", population, " iterations)..."))

            best = set %>% OptTrain()

            Log(c(optimizer, "-", type, " optimization finished (duration = ", Elapsed(watch), " sec, error = ", best$error, ")."))
            if (verbose) print(best$args)
            return(best)
        }

        watch = Log(c("Forecasting ", GetName()))

        set = Dataset()
        best = set %>% TryOptTrain()

        training = Elapsed(watch)
        testing = 0
        updating = 0

        forecast = data.frame()
        for (i in (TestIndexSplitter()+1):Count(set$test)) {
            watch = Stopwatch()
            forecast = Union(forecast, best$model %>% Test(set, i))
            testing = testing + Elapsed(watch)

            watch = Stopwatch()
            best$model = best$model %>% Update(set, i, best$args)
            updating = updating + Elapsed(watch)
        }

        colnames(forecast) = GetName()
        stats = c(colnames(forecast), training, testing, updating)
        names(stats) = c("target", "training", "testing", "updating")
        stats = c(stats, Errors(TestFrom(), forecast))

        return(list(forecast = forecast, stats = t(stats)))
    }

    forecast = data.frame()
    stats = data.frame()
    for (col_y in cols_y) {
        result = Forecast(col_y)
        forecast = Join(forecast, result$forecast)
        stats = Union(stats, result$stats)
        TryExport(forecast, 'Forecast')
        TryExport(stats, 'Stats')
    }

    if (verbose) print(stats)
    return(forecast)
}
