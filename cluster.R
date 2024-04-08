# Load required libraries
library(cluster)
library(cclust)
library(fastcluster)
library(caret)
library(mlbench)
library(Rcmdr)
library(pmml)
library(XML)
library(kohonen)
library(factoextra)
library(magrittr)

# Read data
df <- read.csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00396/Sales_Transactions_Dataset_Weekly.csv')

# Exploratory Data Analysis - Boxplot
par(mfrow=c(2, 3), mar=c(4,4,2,2))  # Layout and margins
for (i in 2:ncol(df)) {
  boxplot(df[, i], main=names(df)[i], col="skyblue", border="black")

}


# Quantile calculation
quant <- function(x) {quantile(x, probs=c(0.95, 0.90, 0.99))}
out1 <- sapply(df[, -1], quant)

# KMeans clustering
sumsq <- NULL
par(mfrow=c(1,2))
for (i in 1:15) sumsq[i] <- sum(kmeans(df[, 56:107], centers=i, iter.max=500, nstart=50)$withinss)
plot(1:15, sumsq, type="b", xlab="Number of Clusters", ylab="Within groups sum of squares", main="Screeplot")

# Cluster analysis
set.seed(121)
km <- kmeans(df[, 56:107], centers=4, nstart=17, iter.max=500)
summary(km)
km$centers
km$withinss

# Predicting clusters
predict.kmeans <- function(km, data) {
  d <- as.matrix(dist(rbind(km$centers, data)))[-(1:nrow(km$centers)), 1:nrow(km$centers)]
  out <- apply(d, 1, which.min)
  return(out)
}
Cluster <- cbind(df[, 56:107], Membership=km$cluster)
Cluster$Predicted <- predict.kmeans(km, df[, 56:107])
table(Cluster$Membership, Cluster$Predicted)

# Writing results to a CSV file
write.csv(Cluster, "predout1.csv")

# Hierarchical clustering
hfit <- hclust(dist(df[, 56:107], method="euclidean"), method="ward.D2")
plot(hfit, hang=-0.005, cex=0.7)

# Dendrogram
plot(hfit)

# Silhouette analysis for hierarchical clustering
sil <- silhouette(cutree(hfit, k=4), daisy(df[, 56:107]))
plot(sil)

# Partitioning Around Medoids (PAM)
pam_res <- pam(df[, 56:107], 6)
plot(silhouette(pam_res))

# Agglomerative hierarchical clustering
ar <- agnes(df[, 56:107])
sil_agg <- silhouette(cutree(ar, k=2), daisy(df[, 56:107]))
plot(sil_agg, nmax=80, cex.names=0.5)

# Model-based clustering
clus <- Mclust(df[, 56:107])
summary(clus)

# Visualization of model-based clustering
plot(clus, data=df[, 56:107], what="BIC")

# Self-organizing maps
som_grid <- somgrid(xdim=20, ydim=20, topo="hexagonal")
som_model <- som(as.matrix(df[, 56:107]))
plot(som_model, type="changes", col="blue")
plot(som_model, type="count")

# Distance analysis
res.dist <- get_dist(df[, 56:107], stand=TRUE, method="pearson")
fviz_dist(res.dist, gradient=list(low="#00AFBB", mid="white", high="#FC4E07"))

# Gap statistic for choosing number of clusters
fviz_nbclust(df[, 56:107], kmeans, method="gap_stat")
fviz_nbclust(df[, 56:107], kmeans, method="silhouette")
fviz_nbclust(df[, 56:107], kmeans, method="wss")

# Clustering results visualization
set.seed(123)
km.res <- kmeans(df[, 56:107], 6, nstart=25)
fviz_cluster(km.res, data=df[, 56:107], ellipse.type="convex", palette="jco")

# Hierarchical clustering visualization
res.hc <- df[, 56:107] %>%
  scale() %>%
  dist(method="euclidean") %>%
  hclust(method="ward.D2")
fviz_dend(res.hc, k=6, cex=0.5, k_colors=c("#2E9FDF", "#00AFBB", "#E7B800", "#FC4E07"), color_labels_by_k=TRUE, rect=TRUE)

# Clustering tendency
gradient.color <- list(low="steelblue", high="white")
df %>% scale() %>% get_clust_tendency(n=50, gradient=gradient.color)

# Number of clusters determination using NbClust
set.seed(123)
res.nbclust <- df %>%
  scale() %>%
  NbClust(distance="euclidean", min.nc=2, max.nc=10, method="complete", index="all")
fviz_nbclust(res.nbclust, ggtheme=theme_minimal())

# Hierarchical K-means
set.seed(123)
res.hc <- df %>%
  scale() %>%
  eclust("hclust", k=6, graph=FALSE)
fviz_dend(res.hc, palette="jco", rect=TRUE, show_labels=FALSE)
fviz_silhouette(res.hc)

# Silhouette widths
sil <- res.hc$silinfo$widths
neg_sil_index <- which(sil[, 'sil_width'] < 0)
sil[neg_sil_index, , drop=FALSE]

# Hierarchical K-means visualization
df <- scale(df)
res.hk <- hkmeans(df, 6)
fviz_dend(res.hk, cex=0.6, palette="jco", rect=TRUE, rect_border="jco", rect_fill=TRUE)
fviz_cluster(res.hk, palette="jco", repel=TRUE, ggtheme=theme_classic())
