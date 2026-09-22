#3. Common Elements in Two Lists
def get_list():
    return list(map(int, input("Enter elements: ").split()))
def find_common(list1, list2):
    common = []
    for i in list1:
        if i in list2 and i not in common:
            common.append(i)
    return common
def print_result(common):
    print("Common elements:", common)
    print("Count:", len(common))
def main():
    print("Enter List 1")
    list1 = get_list()
    print("Enter List 2")
    list2 = get_list()
    common = find_common(list1, list2)
    print_result(common)
main()