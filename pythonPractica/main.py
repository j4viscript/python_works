import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn .inspection import DecisionBoundaryDisplay
import pandas as pd

n_neighbors = 3

#leer el archivo con los datos

datasets = pd.read_csv("dat_clientes.csv")

#caracteristicas 

X = datasets [['ingresos', 'gastos']].values
#clases
y = datasets[['decision']].values

#mapa de colores
cmap_light = ListedColormap(["orange","cyan"])
cmap_bold =["c", "blue"]
for weights in ["uniform", "distance"]:
    knn= neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    knn.fit(X, y)
    #clasifica un nuevo dato
    DatoNuevo = [[0.049170482, 0.689384745]]
    clasifica = knn.predict(DatoNuevo)
    print ("El nuevo dato es de la clasificacion: ", clasifica)
    _,ax=plt.subplots()
    DecisionBoundaryDisplay.from_estimator(knn,X,cmap=cmap_light, response_method="predict", plot_methd="pcolormesh",xlabel="ingresos", ylabel="gastos", shading="auto")
#graficar los puntosa de prueba
sns.scatterplot (
    x=X[:,0],
    y=X[:,1],
    hue = datasets["decision"],
    palette = cmap_bold,
    alpha=1.0,
    edgecolor = "black",)
    
sns.scatterplot (
    x=DatoNuevo[0][0],
    y=DatoNuevo[0][1],
    hue = datasets["decision"],
    palette = cmap_bold,
    alpha=1.0,
    edgecolor = "red",)
plt.title("Clasificador k vecinos + cercanos")
plt.show()