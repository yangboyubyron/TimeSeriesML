#' Optimize: Metaheuristic optimization.
#' Searches for minimum of given cost function.
#' @param f: Cost function (f) that accepts exactly one vector of input values (x): f(x).
#' @param min: Vector of minimum input values, limiting the search space.
#' @param max: Vector of maximum input values, limiting the search space.
#' @param optimizer: One of available metaOpt algorithms.
#' @param population (1, infinity]: Optimizer population size.
#' @param generations (1, infinity]: Maximum number of optimization iterations.
#' @param other: Other optimizer parameters.
#' @return Search result.
Optimize = function(f, min, max, optimizer = "GOA", population = 50, generations = 100, other = list()) {
    range = as.matrix(Union(min, max))
    count = min(Count(min), Count(max))
    param = c(list(numPopulation = max(1, population), maxIter = max(1, generations)), other)
    result = metaOpt(f, rangeVar = range, numVar = count, algorithm = optimizer, control = param)
    return(result$optimumValue)
}
