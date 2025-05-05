const TreeData = {
  start: {
    id: "start",
    question: "What structure does the input primarily use?",
    options: [
      { label: "Sequence-based (e.g., strings, arrays)", target: "sequenceType" },
      { label: "Subset of Items / Weights", target: "subsetType" },
      { label: "Grid / Matrix", target: "gridType" },
      { label: "Tree", target: "treeType" },
      { label: "Graph", target: "graphType" },
      { label: "Bitmask / Assignment", target: "bitmaskType" },
      { label: "Interval / Partitioning", target: "intervalType" },
      { label: "Game Theory / 2-Player", target: "gameTheoryType" },
      { label: "Regex / Edit Distance", target: "regexType" }
    ]
  },

  // === SEQUENCE TYPE ===
  sequenceType: {
    question: "What type of sequence problem is it?",
    options: [
      { label: "Longest Increasing Subsequence (LIS)", target: "LIS" },
      { label: "Longest Common Subsequence (LCS)", target: "LCS" },
      { label: "Longest Common Substring", target: "LCSstr" },
      { label: "Longest Palindromic Substring", target: "palindrome" }
    ]
  },
  LIS: {
    pattern: "Longest Increasing Subsequence",
    description: "Find the longest strictly increasing subsequence in an array.",
    pseudocode: `function LIS(arr):
  n = length of arr
  dp = array of size n, filled with 1
  # dp[i] means the LIS ending at index i
  for i from 1 to n-1:
    for j from 0 to i-1:
      if arr[j] < arr[i]:
        dp[i] = max(dp[i], dp[j] + 1)
  return max(dp)`
  },

  LCS: {
    pattern: "Longest Common Subsequence",
    description: "Find the longest subsequence common to two strings.",
    pseudocode: `function LCS(s1, s2):
  m, n = lengths of s1 and s2
  dp = 2D array of size (m+1) x (n+1)
  for i from 0 to m:
    for j from 0 to n:
      if i == 0 or j == 0:
        dp[i][j] = 0
      else if s1[i-1] == s2[j-1]:
        dp[i][j] = dp[i-1][j-1] + 1
      else:
        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
  return dp[m][n]`
  },

  LCSstr: {
    pattern: "Longest Common Substring",
    description: "Find the longest substring present in both strings.",
    pseudocode: `function LCSstr(s1, s2):
  m, n = lengths of s1 and s2
  dp = 2D array of size (m+1) x (n+1)
  maxLen = 0
  for i from 1 to m:
    for j from 1 to n:
      if s1[i-1] == s2[j-1]:
        dp[i][j] = dp[i-1][j-1] + 1
        maxLen = max(maxLen, dp[i][j])
      else:
        dp[i][j] = 0
  return maxLen`
  },

  palindrome: {
    pattern: "Longest Palindromic Substring",
    description: "Find the longest contiguous substring that is a palindrome.",
    pseudocode: `function longestPalindrome(s):
  n = length of s
  start = 0
  maxLength = 1
  for i from 0 to n:
    # Expand for odd length
    l, r = i, i
    while l >= 0 and r < n and s[l] == s[r]:
      if r - l + 1 > maxLength:
        start = l
        maxLength = r - l + 1
      l -= 1
      r += 1
    # Expand for even length
    l, r = i, i + 1
    while l >= 0 and r < n and s[l] == s[r]:
      if r - l + 1 > maxLength:
        start = l
        maxLength = r - l + 1
      l -= 1
      r += 1
  return s[start:start+maxLength]`
  },

  // === SUBSET / KNAPSACK ===
  subsetType: {
    question: "How many times can each item be used?",
    options: [
      { label: "Only once", target: "boundedKnapsack" },
      { label: "Unlimited times", target: "unboundedKnapsack" }
    ]
  },
  boundedKnapsack: {
    question: "What is your objective?",
    options: [
      { label: "Maximize value within weight", target: "ZeroOneKnapsack" },
      { label: "Subset equals target sum", target: "SubsetSum" },
      { label: "Split into equal subsets", target: "PartitionSubset" },
      { label: "Minimize subset difference", target: "MinSubsetDiff" }
    ]
  },
  unboundedKnapsack: {
    question: "What is your objective?",
    options: [
      { label: "Count combinations", target: "CoinChangeCount" },
      { label: "Minimize coins used", target: "CoinChangeMin" },
      { label: "Maximize value (e.g., Rod Cutting)", target: "RodCutting" }
    ]
  },
  CoinChangeCount: {
    pattern: "Coin Change - Count",
    description: "Count the number of ways to make up the amount using unlimited coins.",
    pseudocode: `function coinChangeCount(coins, target):
  dp = array of size (target+1) with dp[0] = 1
  for coin in coins:
    for amt from coin to target:
      dp[amt] += dp[amt - coin]  # Include coin
  return dp[target]`
  },

  CoinChangeMin: {
    pattern: "Coin Change - Min",
    description: "Find the minimum number of coins needed to make the target amount.",
    pseudocode: `function coinChangeMin(coins, target):
  dp = array of size (target+1), filled with ∞
  dp[0] = 0
  for amt from 1 to target:
    for coin in coins:
      if amt >= coin:
        dp[amt] = min(dp[amt], dp[amt - coin] + 1)
  return dp[target] if dp[target] != ∞ else -1`
  },

  RodCutting: {
    pattern: "Rod Cutting",
    description: "Maximize the value by cutting rod into different lengths.",
    pseudocode: `function rodCutting(prices, n):
  dp = array of size (n+1) with dp[0] = 0
  for i from 1 to n:
    for j from 1 to i:
      dp[i] = max(dp[i], prices[j-1] + dp[i-j])
  return dp[n]`
  },

  ZeroOneKnapsack: {
    pattern: "0/1 Knapsack",
    description: "Select items to maximize value without exceeding weight limit.",
    pseudocode: `function knapsack(weights, values, W):
  n = len(weights)
  dp = array of size (W+1) with all zeros
  for i from 0 to n-1:
    for w from W down to weights[i]:
      dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
  return dp[W]`
  },

  SubsetSum: {
    pattern: "Subset Sum",
    description: "Check if any subset sums to the target.",
    pseudocode: `function subsetSum(nums, target):
  dp = array of size (target+1) with dp[0] = true
  for num in nums:
    for s from target down to num:
      dp[s] |= dp[s - num]
  return dp[target]`
  },

  PartitionSubset: {
    pattern: "Partition Equal Subset",
    description: "Determine if array can be partitioned into two equal-sum subsets.",
    pseudocode: `function canPartition(nums):
  total = sum(nums)
  if total % 2 != 0: return False
  return subsetSum(nums, total // 2)`
  },

  MinSubsetDiff: {
    pattern: "Min Subset Difference",
    description: "Find two subsets such that difference of sums is minimized.",
    pseudocode: `function minSubsetDiff(nums):
  total = sum(nums)
  target = total // 2
  dp = array of size (target+1) with dp[0] = true
  for num in nums:
    for s from target down to num:
      dp[s] |= dp[s - num]
  for s from target down to 0:
    if dp[s]: return total - 2 * s`
  },


  // === GRID ===
  gridType: {
    question: "What is your grid objective?",
    options: [
      { label: "Count unique paths", target: "UniquePaths" },
      { label: "Min path sum", target: "MinPathSum" }
    ]
  },
  UniquePaths: {
    pattern: "Unique Paths",
    description: "Count how many unique paths from top-left to bottom-right.",
    pseudocode: `function uniquePaths(m, n):
  dp = 2D array of size m x n
  for i in 0 to m-1: dp[i][0] = 1
  for j in 0 to n-1: dp[0][j] = 1
  for i in 1 to m-1:
    for j in 1 to n-1:
      dp[i][j] = dp[i-1][j] + dp[i][j-1]
  return dp[m-1][n-1]`
  },

  MinPathSum: {
    pattern: "Minimum Path Sum",
    description: "Find path from top-left to bottom-right with minimum cost.",
    pseudocode: `function minPathSum(grid):
  m, n = dimensions of grid
  dp = copy of grid
  for i from 1 to m-1: dp[i][0] += dp[i-1][0]
  for j from 1 to n-1: dp[0][j] += dp[0][j-1]
  for i from 1 to m-1:
    for j from 1 to n-1:
      dp[i][j] += min(dp[i-1][j], dp[i][j-1])
  return dp[m-1][n-1]`
  },

  // === TREE ===
  treeType: {
    question: "What are you computing on the tree?",
    options: [
      { label: "Max path sum", target: "TreeMaxPath" },
      { label: "Diameter", target: "TreeDiameter" },
      { label: "Max non-adjacent sum (House Robber III)", target: "TreeHouseRobber" }
    ]
  },

  TreeMaxPath: {
    pattern: "Maximum Path Sum in Binary Tree",
    description: "Find the maximum path sum between any two nodes in a binary tree.",
    pseudocode: `function maxPathSum(root):
  max_sum = -Infinity
  function dfs(node):
    if node is null: return 0
    left = max(0, dfs(node.left))
    right = max(0, dfs(node.right))
    max_sum = max(max_sum, left + right + node.val)
    return max(left, right) + node.val
  dfs(root)
  return max_sum`
  },

  TreeDiameter: {
    pattern: "Diameter of Binary Tree",
    description: "Find the longest path between any two nodes in a binary tree.",
    pseudocode: `function diameter(root):
  max_diameter = 0
  function dfs(node):
    if node is null: return 0
    left = dfs(node.left)
    right = dfs(node.right)
    max_diameter = max(max_diameter, left + right)
    return max(left, right) + 1
  dfs(root)
  return max_diameter`
  },

  TreeHouseRobber: {
    pattern: "House Robber III",
    description: "Max sum of non-adjacent nodes in a binary tree.",
    pseudocode: `function rob(root):
  function dfs(node):
    if node is null: return [0, 0]  # [robbed, not_robbed]
    left = dfs(node.left)
    right = dfs(node.right)
    rob = node.val + left[1] + right[1]
    not_rob = max(left) + max(right)
    return [rob, not_rob]
  return max(dfs(root))`
  },


  // === REGEX / EDIT ===
  regexType: {
    question: "Which string pattern problem?",
    options: [
      { label: "Edit Distance", target: "EditDistance" },
      { label: "Regex Matching", target: "RegexMatch" }
    ]
  },
  EditDistance: {
    pattern: "Edit Distance",
    description: "Minimum operations (insert, delete, replace) to convert s1 to s2.",
    pseudocode: `function editDistance(s1, s2):
  m, n = len(s1), len(s2)
  dp = 2D array (m+1 x n+1)
  for i in 0 to m: dp[i][0] = i
  for j in 0 to n: dp[0][j] = j
  for i in 1 to m:
    for j in 1 to n:
      if s1[i-1] == s2[j-1]:
        dp[i][j] = dp[i-1][j-1]
      else:
        dp[i][j] = 1 + min(dp[i-1][j-1], dp[i][j-1], dp[i-1][j])
  return dp[m][n]`
  },

  RegexMatch: {
    pattern: "Regular Expression Matching",
    description: "Match string against pattern containing '.' and '*'.",
    pseudocode: `function isMatch(s, p):
  m, n = len(s), len(p)
  dp = 2D array of (m+1) x (n+1), dp[0][0] = True
  for j from 1 to n:
    if p[j-1] == '*': dp[0][j] = dp[0][j-2]
  for i from 1 to m:
    for j from 1 to n:
      if p[j-1] == s[i-1] or p[j-1] == '.':
        dp[i][j] = dp[i-1][j-1]
      elif p[j-1] == '*':
        dp[i][j] = dp[i][j-2] or (p[j-2] in {s[i-1], '.'} and dp[i-1][j])
  return dp[m][n]`
  },

  // === GAME THEORY ===
  gameTheoryType: {
    question: "What type of game problem is it?",
    options: [
      { label: "Optimal strategy for picking values", target: "TwoPlayerGame" },
      { label: "Stone Game", target: "StoneGame" }
    ]
  },

  TwoPlayerGame: {
    pattern: "Optimal Strategy for Game",
    description: "Max score a player can achieve if both play optimally.",
    pseudocode: `function optimalStrategy(coins):
  n = len(coins)
  dp = 2D array of size n x n
  for i in 0 to n:
    dp[i][i] = coins[i]
  for length in 2 to n:
    for i in 0 to n - length:
      j = i + length - 1
      dp[i][j] = max(coins[i] + min(dp[i+2][j], dp[i+1][j-1]),
                     coins[j] + min(dp[i][j-2], dp[i+1][j-1]))
  return dp[0][n-1]`
  },

  StoneGame: {
    pattern: "Stone Game",
    description: "Determine if the first player can win a stone game where players remove stones from ends.",
    pseudocode: `function stoneGame(piles):
  n = len(piles)
  dp = 2D array of size n x n
  for i in 0 to n:
    dp[i][i] = piles[i]
  for length in 2 to n:
    for i in 0 to n - length:
      j = i + length - 1
      dp[i][j] = max(piles[i] - dp[i+1][j], piles[j] - dp[i][j-1])
  return dp[0][n-1] > 0`
  },


  // === BITMASK ===
  bitmaskType: {
    question: "Which bitmask problem are you solving?",
    options: [
      { label: "Traveling Salesman Problem (TSP)", target: "TSP" },
      { label: "Job Assignment", target: "JobAssignment" },
      { label: "Minimum Incompatibility", target: "MinIncompatibility" }
    ]
  },

  TSP: {
    pattern: "Traveling Salesman Problem",
    description: "Find the shortest route that visits all cities and returns to origin.",
    pseudocode: `function tsp(graph):
  n = len(graph)
  dp = 2D array of size (1 << n) x n, filled with ∞
  dp[1][0] = 0
  for mask in 1 to (1 << n):
    for u in 0 to n:
      if mask & (1 << u):
        for v in 0 to n:
          if not mask & (1 << v):
            dp[mask | (1 << v)][v] = min(dp[mask | (1 << v)][v], dp[mask][u] + graph[u][v])
  return min(dp[(1 << n) - 1][j] + graph[j][0] for j in 0 to n-1)`
  },

  JobAssignment: {
    pattern: "Job Assignment Problem",
    description: "Assign jobs to workers to minimize total cost.",
    pseudocode: `function assignJobs(cost):
  n = len(cost)
  dp = array of size 2^n, filled with ∞
  dp[0] = 0
  for mask in 0 to 2^n:
    k = count of set bits in mask
    for j in 0 to n:
      if not mask & (1 << j):
        dp[mask | (1 << j)] = min(dp[mask | (1 << j)], dp[mask] + cost[k][j])
  return dp[(1 << n) - 1]`
  },
  MinIncompatibility: {
    pattern: "Minimum Incompatibility",
    description: "Partition array into k groups with minimum incompatibility.",
    pseudocode: `# Uses memoization + bitmasking, advanced variant of subset DP` 
  },


  // === INTERVAL ===
  intervalType: {
    question: "What interval-based optimization are you solving?",
    options: [
      { label: "Matrix Chain Multiplication", target: "MatrixChain" },
      { label: "Palindrome Partitioning", target: "PalindromePartition" },
      { label: "Burst Balloons", target: "BurstBalloons" }
    ]
  },

  MatrixChain: {
    pattern: "Matrix Chain Multiplication",
    description: "Determine the most efficient way to multiply chain of matrices.",
    pseudocode: `function matrixChain(p):
  n = len(p) - 1
  dp = 2D array of size n x n
  for l in 2 to n:
    for i in 0 to n-l:
      j = i + l - 1
      dp[i][j] = ∞
      for k in i to j-1:
        dp[i][j] = min(dp[i][j], dp[i][k] + dp[k+1][j] + p[i]*p[k+1]*p[j+1])
  return dp[0][n-1]`
  },

  PalindromePartition: {
    pattern: "Palindrome Partitioning",
    description: "Minimize cuts to partition string into palindromes.",
    pseudocode: `function minCut(s):
  n = len(s)
  isPal = 2D bool array of size n x n
  dp = array of size n, filled with ∞
  for i in 0 to n:
    for j in 0 to i:
      if s[j..i] is palindrome:
        isPal[j][i] = true
        dp[i] = min(dp[i], 0 if j==0 else dp[j-1] + 1)
  return dp[n-1]`
  },

  BurstBalloons: {
    pattern: "Burst Balloons",
    description: "Maximize coins by bursting balloons in optimal order.",
    pseudocode: `function maxCoins(nums):
  add 1 to both ends of nums
  n = len(nums)
  dp = 2D array of size n x n
  for length in 2 to n:
    for left in 0 to n - length:
      right = left + length
      for k in left+1 to right-1:
        dp[left][right] = max(dp[left][right], nums[left]*nums[k]*nums[right] + dp[left][k] + dp[k][right])
  return dp[0][n-1]`
  },


  // === GRAPH ===
  graphType: {
    question: "Which graph-based DP problem?",
    options: [
      { label: "Longest Path in DAG", target: "LongestPathDAG" },
      { label: "Bellman-Ford (Negative Cycles)", target: "BellmanFord" }
    ]
  },

  LongestPathDAG: {
    pattern: "Longest Path in DAG",
    description: "Find the longest path in a Directed Acyclic Graph (DAG).",
    pseudocode: `Use topological sort, then relax edges in topological order`
  },

  BellmanFord: {
    pattern: "Bellman-Ford Algorithm",
    description: "Compute shortest paths allowing negative weights.",
    pseudocode: `function bellmanFord(edges, V, src):
  dist = array of size V, filled with ∞, dist[src] = 0
  for i in 1 to V-1:
    for (u, v, w) in edges:
      if dist[u] + w < dist[v]:
        dist[v] = dist[u] + w
  # Check for negative weight cycle:
  for (u, v, w) in edges:
    if dist[u] + w < dist[v]: return False
  return dist`
  }
};

export default TreeData;
