def main():
    test = [4,3,1,5,2]
    sum = 1
    for i in range(len(test)):
        for j in range(i,len(test)):
            if(i % 2 == 0):
                if(test[i] < test[j]):
                    sum += 1
            else:
                if(test[i] > test[j]):
                    sum -= 1
    print(sum)



if __name__ == '__main__':
    main()
