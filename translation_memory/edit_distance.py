

def edit_distance(string1: str, string2: str):

    n, m = len(string1) + 1, len(string2) + 1

    dp = [[0] * m for _ in range(n)]

    for i in range(1, n):
        dp[i][0] = i

    for j in range(1, m):
        dp[0][j] = j

    for i in range(1, n):
        for j in range(1, m):
            if string1[i-1] == string2[j-1]:
                num = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1])
            else:
                num = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + 1)
            dp[i][j] = num

    return dp[len(string1)][len(string2)]


def main():

    string1 = "hello world"
    string2 = "edit distance"
    print(edit_distance(string1, string2))


if __name__ == "__main__":
    main()
