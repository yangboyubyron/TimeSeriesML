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
#' @param ...: Training parameters specific to forecast model.
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
               verbose = FALSE,
               folder = NULL,
               ...) {
    DS = function(col_y) {
        set = TSFDS(data,
                    col_time,
                    cols_x_required,
                    cols_x_optional,
                    col_y,
                    horizon,
                    lookback,
                    splitter,
                    uncertainty,
                    encoding,
                    normalization)
        return(set)
    }

    GetName = function(col_y) return(ifelse(Count(colnames(data)) == 0, col_y, colnames(data)[col_y]))
    CanSave = function() return(type == 'RNN' && !is.null(folder))
    Save = function(model, col_y) SaveRNN(model, folder, GetName(col_y))
    Load = function(col_y) return(LoadRNN(folder, GetName(col_y)))

    TrySave = function(model, col_y) {
        if (CanSave()) model %>% Save(col_y)
    }
    TryLoad = function(col_y) {
        if (!CanSave()) return(NULL)
        tryCatch({ return(Load(col_y)) }, error = function(e){ return(NULL) })
    }
    TryExport = function(table, file) {
        if (!is.null(folder)) table %>% ExportCSV(folder, name, verbose = verbose)
    }

    Train = function(set) {
        train = set$train
        if (type == 'SVR') {
            return(TrainSVR(train$x, train$y, horizon, lookback, ...))
        } else {
            return(TrainRNN(train$x, train$y, horizon, lookback, ..., verbose = verbose))
        }
    }
    Test = function(model, set, test.index) {
        NormForecast = function(x, y) {
            if (type == 'SVR') {
                return(model %>% TestSVR(x, y, horizon, lookback))
            } else {
                return(model %>% TestRNN(x, y, horizon, lookback))
            }
        }

        test = set$test[[test.index]]
        nres = NormForecast(test$x, test$y)
        return(Norm(nres, set$norm$y, FALSE))
    }
    Update = function(model, set, test.index) {
        if (type == 'SVR') {
            x = set$train$x
            y = set$train$y
            for (i in 1:test.index) {
                test = set$test[[i]]
                rows = (Count(test$x)-horizon+1):Count(test$x)
                x = Union(x, test$x[rows,])
                y = Union(y, test$y[rows,])
            }

            return(TrainSVR(x, y, horizon, lookback, ...))
        } else {
            test = set$test[[test.index]]
            model %>% UpdateRNN(test$x, test$y, horizon, lookback, verbose = verbose)
            return(model)
        }
    }
    Errors = function(col_y, forecast) {
        from = splitter + 1
        to = from + Count(forecast) - 1
        errors = EvaluateForecast(data[from:to, col_y], forecast, verbose = verbose)
        return(errors)
    }

    Forecast = function(col_y) {
        target = GetName(col_y)
        Log(c("Forecasting ", target))

        watch = StopwatchStartNew()
        set = DS(col_y)

        model = TryLoad(col_y)
        if (is.null(model)) {
            model = Train(set) # TODO: Optimize.
            model %>% TrySave(col_y)
        }

        training = StopwatchElapsedSeconds(watch)
        testing = 0
        updating = 0

        forecast = data.frame()
        for (i in 1:Count(set$test)) {
            watch = StopwatchStartNew()
            forecast = Union(forecast, model %>% Test(set, i))
            testing = testing + StopwatchElapsedSeconds(watch)

            watch = StopwatchStartNew()
            model = model %>% Update(set, i)
            updating = updating + StopwatchElapsedSeconds(watch)

            model %>% TrySave(col_y)
        }

        colnames(forecast) = target

        stats = c(target, training, testing, updating)
        names(stats) = c("target", "training", "testing", "updating")
        stats = c(stats, Errors(col_y, forecast))

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
