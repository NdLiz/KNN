class CustomerData:
    def __init__(self, height, weight, shirt_size=None):
        #Inicializa un objeto CustomerData con altura, peso y talla de camiseta.
        self.height = height
        self.weight = weight
        self.shirt_size = shirt_size
    
    def compute_distance(self, other):
        #Calcula la distancia euclidiana entre este punto y otro punto de datos.
        return ((self.height - other.height) ** 2 + (self.weight - other.weight) ** 2) ** 0.5

class KNNModel:
    def __init__(self, k):
        #Inicializa el modelo KNN con un número específico de vecinos (k).
        self.k = k
        self.customers = []

    def train(self, customers):
        #Entrena el modelo con una lista de datos de clientes.
        
        self.customers = customers

    def predict(self, new_customer):
        # Predice la talla de camiseta para un nuevo cliente basado en el modelo entrenado.
        # Calcula las distancias entre el nuevo cliente y todos los clientes en el conjunto de datos
        distances = [(customer.compute_distance(new_customer), customer) for customer in self.customers]
        
        # Ordena las distancias en orden ascendente
        distances.sort(key=lambda x: x[0])
        
        # Obtiene los k vecinos más cercanos
        nearest_neighbors = [customer for _, customer in distances[:self.k]]
        
        # Determina la talla de camiseta más común entre los vecinos más cercanos
        return self._determine_size(nearest_neighbors)

    def _determine_size(self, neighbors):
        #Determina la talla de camiseta más común entre los vecinos más cercanos.
        
        size_counts = {}
        for neighbor in neighbors:
            if neighbor.shirt_size in size_counts:
                size_counts[neighbor.shirt_size] += 1
            else:
                size_counts[neighbor.shirt_size] = 1
        # Devuelve la talla de camiseta con la mayor frecuencia
        return max(size_counts, key=size_counts.get)

class DataSet:
    def __init__(self, heights, weights, shirt_sizes):
        # Inicializa el repositorio de datos con listas de alturas, pesos y tallas de camisetas.
        # Crea una lista de objetos CustomerData a partir de las listas de alturas, pesos y tallas de camisetas
        self.customers = [CustomerData(h, w, s) for h, w, s in zip(heights, weights, shirt_sizes)]
    
    def get_customers(self):
        #Devuelve la lista de objetos CustomerData.
    
        return self.customers

# Datos del dataset
heights = [158, 158, 158, 160, 160, 163, 163, 160, 163, 165, 165, 165, 168, 168, 168, 170, 170, 170]
weights = [58, 59, 63, 59, 60, 60, 61, 64, 64, 61, 62, 65, 62, 63, 66, 63, 64, 68]
shirt_sizes = ['M', 'M', 'M', 'M', 'M', 'M', 'M', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L']

# Crear y cargar el repositorio de datos
repository = DataSet(heights, weights, shirt_sizes)

# Crear el modelo KNN con k=3
knn_model = KNNModel(k=3)
# Entrenar el modelo con los datos del repositorio
knn_model.train(repository.get_customers())

# Crear un nuevo cliente con altura 170 y peso 70
clienteN= CustomerData(163, 58)

# Predecir la talla de camiseta para el nuevo cliente
size = knn_model.predict(clienteN)

# Imprimir la talla de camiseta predicha para el nuevo cliente
print(f'\nSe ha predicho la talla del cliente, su talla es: {size}\n')
