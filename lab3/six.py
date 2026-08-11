from scipy.spatial.distance import minkowski
from four import minkowski_distance
v1 = [2, 4, 6]
v2 = [1, 2, 3]
my_result = minkowski_distance(v1, v2, 2)
scipy_result = minkowski(v1, v2, 2)
print("My function result   :", my_result)
print("Scipy function result:", scipy_result)
if round(my_result, 6) == round(scipy_result, 6):
    print("matching")
else:
    print("not matching")