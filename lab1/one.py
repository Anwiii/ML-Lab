# 1. Vowels and Consonants
def count_vowels_consonants(s):
    s = s.lower()
    vowels = 0
    consonants = 0
    for c in s:
        if c in "aeiou":
            vowels += 1
        elif c.isalpha():
            consonants += 1

    return vowels, consonants
def main():
    s = input("Enter string: ")
    v, c = count_vowels_consonants(s)
    print(f"Vowels: {v}, Consonants: {c}")

main()