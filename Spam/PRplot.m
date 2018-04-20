precision = [0.666 0.707 0.907 0.918 0.939 0.957 0.962 0.965 0.972 0.981 0.985 0.987 ...
    0.986 0.986 0.986 0.985 0.985 0.985 0.984 0.988 0.992 0.998 ...
    0.998 0.998 1 0.997 0];
recall = [1 0.994 0.990 0.989 0.988 0.982 0.977 0.970 0.954 0.901 0.838 0.749 ...
    0.642 0.564 0.508 0.443 0.412 0.354 0.313 0.286 0.245 0.223 ...
    0.196 0.1 0.08 0.036 0];
precision = fliplr(precision);
recall = fliplr(recall);
plot(recall, precision, 'b');