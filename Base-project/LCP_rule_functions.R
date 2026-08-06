## Sheet containing functions that calculate the different LCP rules

## Functions for repeats

repeatn <- function(x, y, n) {
  # Function for event repeated n times

  # Create empty output array
  y_output <- array(0, dim(y))
  for (i in seq_len(dim(y)[1])) {
    # for each individual in cohort
    # set count to zero
    count <- 0
    for (j in seq_len(dim(y)[2])) {
      # for each time point
      if (x[i, 9, j] == 1) {
        # check if they have an x9 event
        count <- count + 1
      } else {
        count <- 0
      }
      if (count == n) {
        # count will be equal to the number of events in a row
        # if count >= n then that individual gets a positive outcome
        y_output[i, dim(y)[2]] <- 1
      }
    }
  }
  return(y_output)
}

repeat2 <- function(x, y) {
  return(repeatn(x, y, 2))
}

repeat3 <- function(x, y) {
  return(repeatn(x, y, 3))
}

repeat4 <- function(x, y) {
  return(repeatn(x, y, 4))
}

repeat2_2 <- function(x, y) {
  # function for two events repeated twice

  # create empty output array
  y_output <- array(0, dim(y))
  for (i in seq_len(dim(y)[1])) {
    # for each individual in cohort
    # set counts to zero
    count1 <- 0
    count2 <- 0
    mark <- 0
    outj <- 0
    for (j in seq_len(dim(y)[2])) {
      # for each time point
      if (x[i, 9, j] == 1) {
        # check if they have an x9 event
        count1 <- count1 + 1
      } else {
        count1 <- 0
      }
      if (count1 == 2) {
        # if have a repeat of 2 then increase mark and note time
        mark <- 1
        outj <- j
      }
    }
    if (mark == 1 && outj < dim(y)[2]) {
      # if have at least one repeat
      for (k in (outj + 1):dim(y)[2]) {
        # search rest of time frame
        if (x[i, 9, k] == 1) {
          count2 <- count2 + 1
        } else {
          count2 <- 0
        }
        if (count2 == 2) {
          # if have two repeats of two get positive outcome
          y_output[i, dim(y)[2]] <- 1
        }
      }
    }
  }
  return(y_output)
}

################################################################################
## Functions for order
order1 <- function(x, y) {
  y_output <- array(0, dim(y))
  for (i in seq_len(dim(y)[1])) {
    time3 <- c()
    time4 <- c()
    for (j in seq_len(dim(y)[2])) {
      if (x[i, 8, j] == 1) {
        time3 <- c(time3, j)
      }
      if (x[i, 9, j] == 1) {
        time4 <- c(time4, j)
      }
    }
    if (!(is.null(time3) || is.null(time4))) {
      if (time3[1] < time4[1]) {
        y_output[i, dim(y)[2]] <- 1
      }
    }
  }
  return(y_output)
}

order2 <- function(x, y) {
  y_output <- array(0, dim(y))
  for (i in seq_len(dim(y)[1])) {
    time3 <- c()
    time4 <- c()
    for (j in seq_len(dim(y)[2])) {
      if (x[i, 8, j] == 1) {
        time3 <- c(time3, j)
      }
      if (x[i, 9, j] == 1) {
        time4 <- c(time4, j)
      }
    }
    if (!(is.null(time3) || is.null(time4)) && length(time3) > 1) {
      if (time3[1] < time4[1] && time3[2] < time4[1]) {
        y_output[i, dim(y)[2]] <- 1
      }
    }
  }
  return(y_output)
}

order3 <- function(x, y) {
  y_output <- array(0, dim(y))
  for (i in seq_len(dim(y)[1])) {
    time3 <- c()
    time4 <- c()
    for (j in seq_len(dim(y)[2])) {
      if (x[i, 8, j] == 1) {
        time3 <- c(time3, j)
      }
      if (x[i, 9, j] == 1) {
        time4 <- c(time4, j)
      }
    }
    if (!(is.null(time3) || is.null(time4)) && length(time3) > 2) {
      if (time3[1] < time4[1] && time3[2] < time4[1] && time3[3] < time4[1]) {
        y_output[i, dim(y)[2]] <- 1
      }
    }
  }
  return(y_output)
}

order4 <- function(x, y) {
  y_output <- array(0, dim(y))
  for (i in seq_len(dim(y)[1])) {
    time3 <- c()
    time4 <- c()
    for (j in seq_len(dim(y)[2])) {
      if (x[i, 8, j] == 1) {
        time3 <- c(time3, j)
      }
      if (x[i, 9, j] == 1) {
        time4 <- c(time4, j)
      }
    }
    if (!(is.null(time3) || is.null(time4)) && length(time3) > 3) {
      if (time3[1] < time4[1] && time3[2] < time4[1] && time3[3] < time4[1] && time3[4] < time4[1]) {
        y_output[i, dim(y)[2]] <- 1
      }
    }
  }
  return(y_output)
}

################################################################################
## Functions for timing
timing_nevent_myear <- function(x, y, n, m) {
  y_output <- array(0, dim(y))
  for (i in seq_len(dim(y)[1])) {
    time3 <- c()
    time4 <- c()
    for (j in seq_len(dim(y)[2])) {
      if (x[i, 8, j] == 1) {
        time3 <- c(time3, j)
      }
      if (x[i, 9, j] == 1) {
        time4 <- c(time4, j)
      }
    }
    if (!(is.null(time3) || is.null(time4))) {
      test <- 0
      for (q in seq_along(time3)) {
        if (any(abs(time4 - time3[q]) <= m)) {
          test <- test + 1
        }
      }
      if (test >= n) {
        y_output[i, dim(y)[2]] <- 1
      }
    }
  }
  return(y_output)
}

timing1_0 <- function(x, y) {
  return(timing_nevent_myear(x, y, 1, 0))
}

timing2_0 <- function(x, y) {
  return(timing_nevent_myear(x, y, 2, 0))
}

timing1_1 <- function(x, y) {
  return(timing_nevent_myear(x, y, 1, 1))
}

timing2_1 <- function(x, y) {
  return(timing_nevent_myear(x, y, 2, 1))
}

timing3_1 <- function(x, y) {
  return(timing_nevent_myear(x, y, 3, 1))
}

timing4_1 <- function(x, y) {
  return(timing_nevent_myear(x, y, 4, 1))
}

timing1_2 <- function(x, y) {
  return(timing_nevent_myear(x, y, 1, 2))
}

timing2_2 <- function(x, y) {
  return(timing_nevent_myear(x, y, 2, 2))
}

timing3_2 <- function(x, y) {
  return(timing_nevent_myear(x, y, 3, 2))
}

timing4_2 <- function(x, y) {
  return(timing_nevent_myear(x, y, 4, 2))
}

timing1_3 <- function(x, y) {
  return(timing_nevent_myear(x, y, 1, 3))
}

timing2_3 <- function(x, y) {
  return(timing_nevent_myear(x, y, 2, 3))
}

timing3_3 <- function(x, y) {
  return(timing_nevent_myear(x, y, 3, 3))
}

timing4_3 <- function(x, y) {
  return(timing_nevent_myear(x, y, 4, 3))
}

timing1_4 <- function(x, y) {
  return(timing_nevent_myear(x, y, 1, 4))
}

timing2_4 <- function(x, y) {
  return(timing_nevent_myear(x, y, 2, 4))
}

timing3_4 <- function(x, y) {
  return(timing_nevent_myear(x, y, 3, 4))
}

timing4_4 <- function(x, y) {
  return(timing_nevent_myear(x, y, 4, 4))
}

################################################################################
## Functions for period
criticaln <- function(x, y, n) {
  y_output <- array(0, dim(y))
  for (i in seq_len(dim(y)[1])) {
    if (sum(x[i, 9, 1:n]) == 0) {
      y_output[i, dim(y)[2]] <- 1
    }
  }
  return(y_output)
}

critical5 <- function(x, y) {
  return(criticaln(x, y, 5))
}

critical10 <- function(x, y) {
  return(criticaln(x, y, 10))
}

critical15 <- function(x, y) {
  return(criticaln(x, y, 15))
}

critical20 <- function(x, y) {
  return(criticaln(x, y, 20))
}

critical25 <- function(x, y) {
  return(criticaln(x, y, 25))
}

critical30 <- function(x, y) {
  return(criticaln(x, y, 30))
}

critical35 <- function(x, y) {
  return(criticaln(x, y, 35))
}

sensitiven <- function(x, y, n) {
  y_output <- array(0, dim(y))
  ##make my own y_output
  for (i in seq_len(dim(y)[1])) {
    if (sum(x[i, 9, 1:10]) >= n) {
      y_output[i, dim(y)[2]] <- 1
    }
  }
  return(y_output)
}

sensitive1 <- function(x, y) {
  return(sensitiven(x, y, 1))
}

sensitive2 <- function(x, y) {
  return(sensitiven(x, y, 2))
}

sensitive3 <- function(x, y) {
  return(sensitiven(x, y, 3))
}

sensitive4 <- function(x, y) {
  return(sensitiven(x, y, 4))
}

sensitive5 <- function(x, y) {
  return(sensitiven(x, y, 5))
}

weightedn <- function(x, y, n) {
  y_output <- array(0, dim(y))
  for (i in seq_len(dim(y)[1])) {
    count_todler <- sum(x[i, 9, 1:4])
    count_child <- sum(x[i, 9, 5:12])
    count_teen <- sum(x[i, 9, 13:17])
    count_younga <- sum(x[i, 9, 18:25])
    count_adult <- sum(x[i, 9, 26:40])
    if ((count_todler + count_child / 2 + count_teen / 3 + count_younga / 5 + count_adult / 10) > n) {
      y_output[i, dim(y)[2]] <- 1
    }
  }
  return(y_output)
}

weighted2 <- function(x, y) {
  return(weightedn(x, y, 2))
}

weighted2_5 <- function(x, y) {
  return(weightedn(x, y, 2.5))
}

weighted3 <- function(x, y) {
  return(weightedn(x, y, 3))
}

weighted4 <- function(x, y) {
  return(weightedn(x, y, 4))
}

weighted5 <- function(x, y) {
  return(weightedn(x, y, 5))
}