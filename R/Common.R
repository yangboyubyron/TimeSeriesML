#' Now: Returns current time as string.
#' @return A string representing current time.
Now = function() return(paste(Sys.time(), sep = ""))

#' Log: Logs current time and given message.
#' @param message: A single object, vector or a list of objects to be concatenated into a log message.
Log = function(message) {
    log.prefix = paste("[", Now(), "]", sep = "")
    log.message = paste(message, collapse = "")
    write(paste(log.prefix, log.message), stdout())
}

#' StopwatchStartNew: Creates and starts new stopwatch.
#' @return A number representing stopwatch.
StopwatchStartNew = function() return(proc.time())

#' StopwatchElapsedSeconds: Extracts elapsed number of seconds from given stopwatch.
#' @param stopwatch: Stopwatch created with the StopwatchStartNew function.
#' @return Elapsed number of seconds.
StopwatchElapsedSeconds = function(stopwatch) return(as.integer((proc.time() - stopwatch)[3]))

#' Limit: Limits value to [min, max] range.
#' @param value: Numeric value.
#' @param min: Min value.
#' @param max: Max value.
#' @return Limited value.
Limit = function(value, min = NA, max = NA) {
    if (!is.na(min) && value < min) value = min
    if (!is.na(max) && value > max) value = max
    return(value)
}

#' DateTime: Creates DateTime values.
#' @param strings: Strings in DateTime format.
#' @param format: DateTime format (check ?strptime for more details).
#' @return DateTime values.
DateTime = function(strings, format = '%F %T') {
    s = gsub(" ", "-", strings)
    f = gsub(" ", "-", format)
    return(as_datetime(s, format = f))
}

#' DatePart: Extracts DateTime parts.
#' @param timestamps: DateTime values.
#' @param part: DateTime part in DateTime format (check ?strptime for more details).
#' @return DateTime parts.
DatePart = function(timestamps, part) return(as.numeric(format(timestamps, part)))

#' Count: Counts the table rows or collection items.
#' @param data: Table (matrix or data frame) or collection (vector or list).
#' @return The number of table rows or collection length.
Count = function(data) {
    if (is.null(data)) return(0)
    if (is.matrix(data) || is.data.frame(data)) return(nrow(data))
    if (is.vector(data) || is.list(data)) return(length(data))
    return(0)
}

#' Join: Joins columns.
#' @param columns1: Data frame, matrix or verctor with first set of columns.
#' @param columns2: Data frame, matrix or verctor with second set of columns.
#' @param colnames1: String or vector with names of columns in the first set.
#' @param colnames2: String or vector with names of columns in the second set.
#' @param rownames: String or vector with names of rows in the resulting table.
#' @return New table with given columns.
Join = function(columns1, columns2, colnames1 = NULL, colnames2 = NULL, rownames = NULL) {
    Bind = function(table1, table2) {
        if (Count(table1) == 0) return(table2)
        if (Count(table2) == 0) return(table1)
        return(as.data.frame(cbind(table1, table2)))
    }

    table1 = as.data.frame(columns1)
    table2 = as.data.frame(columns2)
    table = Bind(table1, table2)

    if (Count(colnames1) > 0) colnames(table)[1:ncol(table1)] = colnames1
    if (Count(colnames2) > 0) colnames(table)[(ncol(table1) + 1):ncol(table)] = colnames2
    if (Count(rownames) > 0) row.names(table) = rownames

    return(table)
}

#' Union: Creates a union of rows.
#' @param rows1: Data frame, matrix or verctor with first set of rows.
#' @param rows2: Data frame, matrix or verctor with second set of rows.
#' @param rownames1: String or vector with names of rows in the first set.
#' @param rownames2: String or vector with names of rows in the second set.
#' @param colnames: String or vector with names of columns in the resulting table.
#' @return New table with given rows.
Union = function(rows1, rows2, colnames = NULL, rownames = NULL) {
    Bind = function(table1, table2) {
        if (Count(table1) == 0) return(table2)
        if (Count(table2) == 0) return(table1)
        return(as.data.frame(rbind(table1, table2)))
    }

    table = Bind(rows1, rows2)

    if (Count(colnames) > 0) colnames(table) = colnames
    if (Count(rownames) > 0) row.names(table) = rownames

    return(table)
}

#' GetPath: Composes allowed file path.
#' @param folder: Folder name.
#' @param file: File name.
#' @param extension: File extension.
#' @return Composed file path.
GetPath = function(folder, file, extension) {
    file = gsub("[[:punct:]]", "-", paste(file, collapse = ""))
    path = paste(paste(folder, collapse = ""), "\\", file, ".", extension, sep = "")
    return(path)
}

#' ImportCSV: Imports CSV file to a table.
#' @param folder: Export folder name.
#' @param file: Export file name.
#' @param ext: File extension.
#' @param separator: Column separator.
#' @param rows: Shows whether or not to export row names.
#' @param verbose: Shows whether or not to log file path.
#' @return Imported csv file as a table.
ImportCSV = function(folder, file, ext = "csv", separator = "|", rows = FALSE, verbose = FALSE) {
    path = GetPath(folder, file, ext)
    if (verbose) Log(c("Importing ", path))
    if (rows) return(read.table(file = path, sep = separator, header = TRUE, row.names = 1))
    return(read.table(file = path, sep = separator, header = TRUE))
}

#' ExportCSV: Exports table to a CSV file.
#' @param table: Table to be exported.
#' @param folder: Export folder name.
#' @param file: Export file name.
#' @param ext: File extension.
#' @param separator: Column separator.
#' @param rows: Shows whether or not to export row names.
#' @param verbose: Shows whether or not to log file path.
ExportCSV = function(table, folder, file, ext = "csv", separator = "|", rows = FALSE, verbose = FALSE) {
    path = GetPath(folder, file, ext)
    if (verbose) Log(c("Exporting ", path))
    dir.create(paste(folder, collapse = ""), recursive = TRUE, showWarnings = FALSE)
    write.table(table, file = path, sep = separator, col.names = TRUE, row.names = rows, quote = FALSE)
}

#' OpenSqlConnection: Opens trusted connection to SQL Server database.
#' @param instance: SQL Server instance.
#' @param database: SQL Server database.
#' @param verbose: Shows whether or not to log connection details.
#' @return An open connection to SQL Server database.
OpenSqlConnection = function(instance, database, verbose = FALSE) {
    string = paste("driver={SQL Server};server=.\\", instance, ";database=", database, ";trusted_connection=true", sep = "")
    if (verbose) Log(c("Connecting to SQL Server: ", string))

    connection = RODBC::odbcDriverConnect(string)
    if (connection == -1) {
        stop("Failed to connect to SQL Server.")
    } else {
        if (verbose) Log("Connected to SQL Server.")
    }

    return(connection)
}
