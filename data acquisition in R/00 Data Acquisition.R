library(data.table)
library(jsonlite)

if (rstudioapi::isAvailable()){ # if RStudio is active 
        setwd(dirname(rstudioapi::getActiveDocumentContext()$path)) # set directory to current document path
}

dt <- data.table(fromJSON("mpd.slice.0-999.json", flatten = T)$playlists)
dt$tracks[1][[1]]$track_name
tracks <- dt$tracks
playlists <- dt$pid
tracks <- unique(unlist(lapply(tracks, FUN = function(x){paste(x$artist_name, "-", x$track_name)})))
length(dt$tracks[1][[1]]$track_name)


