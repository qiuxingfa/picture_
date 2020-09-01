#  剑指offer

## 第1章 面试的流程

### 1.1 面试官谈面试

* 考察算法和数据结构
* 对公司近况、项目有所了解，对应聘的工作很有热情
* 和面试官多做沟通，做一些整体的设计和规划

### 1.2 面试的3种形式

* 电话面试
  * 不确定面试官问题时，大但向面试官提问
* 远程面试
  * 思考清楚再开始编码
  * 良好的代码命名和缩进
  * 能够进行单元测试
  * 主要是编程习惯和调试能力
* 现场面试

### 1.3 面试的3个环节

* 行为面试
  * 30秒到1分钟介绍自己的主要学习工作经历
  * 项目经历：简短的项目背景、自己完成的任务、为完成任务做了哪些工作、自己的贡献
* 技术面试
  * 基础知识：编程语言、数据结构、算法
  * 高质量的代码
  * 清晰的思路
  * 优化效率的能力
  * 优秀的综合能力

* 应聘者提问

## 第2章 面试需要的基础知识

### 2.2 编程语言

### 2.3 数据结构

1. 赋值运算符函数
2. 实现Singleleton模式
3. 找出数组中的重复数字
   * 排序扫描

   * 哈希表

   * 与下标进行对比
   * 不修改数组找出重复的数字
      * 逐一把原数组的数字复制到辅助数组中
      * 类似二分法，判断值为1-m的数字数目是否大于m
4. 二维数组中的查找

   * 选取数组右上角（左下角）的数字进行比较，删除所在列或行
5. 替换空格

   * 扫描，替换
   * 统计空格数，从后面开始复制和替换
6. 从尾到头打印链表
7. 重建二叉树
   * 递归地构建左右子树，前序遍历的第一个节点为根节点，中序遍历的根节点左边为左子树，右边为右子树
8. 二叉树的下一个节点
   * 分情况分析
9. 用两个栈实现队列
   * push时插入栈A，pop时将栈A数据插入栈B然后输出栈删除的第一个数

### 2.4 算法和数据操作

10. 斐波那契数列
    * 根据函数定直接写出的函数存在很多重复的计算
    * 从下往上计算，根据f(n-2)和f(n-1)计算f(n)
    * 根据数学公式进行矩阵运算（比较复杂，不太实用）
    * 青蛙跳台问题：一只青蛙一次可以跳上1级台阶，也可以跳上2级台阶。求该青蛙跳上一个n级台阶共有多少种跳法。（扩展：青蛙一次可以有n种跳法，f(n)=2^(n-1)）
    * 用2×1的小矩形横着或者竖着去覆盖更大的矩形。请问用8个2×1的小矩形无重复地覆盖一个2×8的大矩形，总共有多少种方法
11. 旋转数组的最小数字
    * 二分查找法，用两个指针指向首尾，并和中间元素进行比较
12. 矩阵中的路径
    * 回溯法
13. 机器人的运动范围
    * 回溯法
14. 剪绳子
    * 动态规划，将问题分解为小问题
    * 贪婪算法，n>=5时，尽可能地多剪长度为3的绳子，当剩下的绳子长度为4时，把绳子减成两端长度为2的绳子，证明：3(n-3) >= 2(n-2)
15. 二进制中1的个数
    * 常规解法，首先把n 和1 做与运算， 判断n 的最低位是不是为1 。接着把1 左移一位得到2, 再和n 做与运算， 就能判断n 的次低位是不是1 … 这样反复左移，每次都能判断n 的其中一位是不是1。
    * 把一个整数减去1 ，再和原整数做与运算， 会把该整数最右边的1变成0。那么一个整数的二进制表示中有多少个1,就可以进行多少次这样的操作。

## 第3章 高质量的代码

### 3.2 代码的规范性

* 清晰的书写，清晰的不具，合理的命名

### 3.3 代码的完整性

* 把尽可能的输入都想清楚，考虑各种边界值的测试用例，考虑各种可能的错误输入
* 错误处理方法：
  * 函数用返回值来告知调用者是否出错
  * 当错误发生时设置一个全局变量
  * 异常

16. 数值的整数次方
    * 分情况讨论
    * 按指数为奇数偶数的情况分，用递归的方法实现
17. 打印从1到最大的n位数
    * 可能存在大数问题
    * 用字符串表示数字
18. 删除链表的节点
    * 遍历查找要删除的节点
    * 先把i节点的下一个节点j的内容复制到i，然后把i的指针指向节点j的下一个节点，此时再删除节点j，如果要删除的节点位于链表的尾部，那么就需要进行顺序遍历
    * 删除链表中的重复节点
19. 正则表达式匹配
    * 当模式为'.'或者字符和模式相互匹配时，则接着匹配后面的字符，当模式中第二个字符为'*'时，分析匹配方式的不同情况
20. 表示数值的字符串
    * 表示数值的字符串遵循模式A[.\[B\]][e|EC] 或者B[e|EC] ，其中A 为数值的整数部分， B紧跟着小数点为数值的小数部分，C紧跟着'e'或者'E'为数值的指数部分。
21. 调整数据顺序使奇数位于偶数前面
    * 不考虑时间复杂度，扫描数组，每碰到一个偶数，将后面的数字向前移一位，将改数放到最后
    * 维护两个指针，第一个指针指向数组的第一个数字，它只向后移动，第二个指针指向数组的最后一个数字，它只向前移动

### 3.4 代码的鲁棒性

22. 链表中倒数第k个节点
    * 遍历两次，第一次得到节点数n，第二次得到前进的步数n-k+1
    * 定义两个指针，当第一个指针指向第k个节点时，启动第二个指针，所以当第二个指针指向结尾时，第一个指针即为倒数第k个节点，要特别注意输入异常的情况
    * 求链表的中间节点。定义两个指针，第一个指针走一步第二个指针走两步。
23. 链表中环的入口节点
    * 首先用两个指针得到环中节点的数目k，再用两个相隔k的指针得到环的入口
24. 反转链表
    * 定义前一节点，当前节点，后一节点三个指针
25. 合并两个排序的链表
    * 使用递归方法，充分考虑两个链表的不同情况
26. 树的子结构
    * 使用递归的方法遍历

## 第4章 解决面试题的思路

### 4.2  画图让抽象问题形象化

27. 二叉树的镜像
    * 先前序遍历这棵树的每个节点，如果遍历到的节点有子节点，就交换它的两个子节点，当交换完所有非叶子节点的左右子节点之后，就得到了树的镜像
28. 对称的二叉树
    * 通过比较二叉树的前序遍历和对称前序遍历是否相同来判断
29. 顺时针打印矩阵
    * 让循环继续的条件是columns>startX\*2 并且 rows > startY\*2，仔细分析打印时每一步的前提条件

### 4.3 举例让抽象问题具体化

30. 包含min函数的栈
    * 使用另一个栈保存最小值
31. 栈的压入、弹出序列
    * 如果下一个弹出的数字刚好是栈顶数字，那么直接弹出，如果下一个弹出的数字不在栈顶，则把压栈序列中还没有入栈的数字压入辅助栈，直到把下一个需要弹出的数字压入栈顶为止，如果所有数字都压入栈后仍没有找到下一个弹出的数字，那么该序列不可能是一个弹出序列
32. 从上到下打印二叉树
    * 每打印一个节点时，把该节点的子节点放入队列中，直至队列中所有节点都被打印
    * 分行从上到下打印二叉树，需保存当前层还没有打印的节点数和下一层的节点数
    * 之字形打印二叉树，打印某一层节点时，把下一层的子节点保存到相应的栈里，如果当前打印的是奇数层，则先保存左子节点，再保存右子节点到以一个栈，如果当前打印的是偶数层，则先保存右子节点再保存左子节点到第二个栈
33. 二叉搜索树的后序遍历序列
    * 最后一个数字是根节点，前面小于根节点的为左子树，后面大于根节点的为右子树，顺序不能乱，用递归
34. 二叉树中和为某一值的路径
    * 前序遍历到某一节点时，把该节点添加到路径上，并累加该节点的值，如果该节点为叶节点，并且路径中节点值的和刚好等于输入的整数，则当前路径符合要求，当前节点访问结束后，自动回到它的父节点

### 4.4 分问题让复杂问题简单化

35. 复杂链表的复制
    * 第一步将每个节点N链接到复制节点N‘，第二步复制节点的随机链接，然后把长链分成两个链表，偶数节点就是复制出来的链表







---

# Leetcode

### 1. 两数之和

* 先排序，然后首位递进查找，时间复杂度为O(nlogn)
* 用字典搜索，时间复杂度为O(n)

~~~python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        dic = {}
        for i,num in enumerate(nums):
            another = target - num
            if another in dic.keys():
                return [dic[another],i]
            dic[num] = i
~~~



### 2. 两数相加

* 使用变量来跟踪进位，考虑两个数长短不一的情况，考虑两个数相加后位数大于原来的数的情况

~~~python
class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        ad = 0
        init = ListNode(0)
        x = init
        while(l1 or l2):
            a = l1.val if l1 else 0
            b = l2.val if l2 else 0
            s = ad + a + b
            num = s%10
            
            ad = 1 if s >= 10 else 0
            init.next = ListNode(num)
            init = init.next
            l1 = l1.next if l1 else None
            l2 = l2.next if l2 else None

        if ad == 1:
            init.next = ListNode(1)
        return x.next
~~~



### 3. 无重复字符的最长字串

* 定义字符到索引的映射，当找到重复字符时，立即跳过该窗口，时间复杂度O(n),空间复杂度O(min(m,n))
* 用两个指针分别记录无重复字串的左右端点

~~~python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        if not s:
            return 0
        st = {}
        i, num = 0, 0
        for j in range(len(s)):
            if s[j] in st:
                i = max(st[s[j]],i)  # 截至j，以j为最后一个元素的最长不重复子串的起始位置
            num = max(num, j - i + 1)
            st[s[j]] = j + 1
        return num
~~~



### 4. 寻找两个有序数组的中位数

* 二分法，时间复杂度O(log(m+n))

~~~python
class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        if len(nums1) > len(nums2):
            nums1, nums2 = nums2, nums1
        m, n = len(nums1), len(nums2)
        left, right = 0, m
        left_max, right_min = 0, 0
        
        while left <= right:
            i = (left+right)//2
            j = (m+n+1)//2 - i
            
            nums_im1 = -2**40 if i==0 else nums1[i-1]
            nums_jm1 = -2**40 if j==0 else nums2[j-1]
            nums_i = 2**40 if i==m else nums1[i]
            nums_j = 2**40 if j==n else nums2[j]
            
            if nums_im1 <= nums_j:
                left_max, right_min = max(nums_im1,nums_jm1), min(nums_i,nums_j)
                left = i+1
            else:
                right = i-1
                
        if (m+n)%2==0:
            return (left_max+right_min)/2
        return left_max
~~~



### 5. 最长回文字符串

* 动态规划-中心扩散法

~~~python
class Solution:
    
    def spread(self, s, left, right):

        while left >= 0 and right < len(s) and s[left]==s[right]:
            left -= 1
            right += 1
        return s[left+1:right]
            
    
    def longestPalindrome(self, s: str) -> str:
        if s == s[::-1]:
            return s
        res = [s[0]]
        for i in range(len(s)):
            odd = self.spread(s,i,i)
            even = self.spread(s,i,i+1)
            res = max(odd,even,res,key=len)
        return res
~~~



### 6. Z字形变换

* 从左到右迭代s，将每个字符添加到合适的行

~~~python
class Solution:
    def convert(self, s, numRows):
        if len(s) <= numRows or numRows == 1:
            return s
        
        res = ['' for i in range(numRows)]
        i,flag = 0, 1
        for c in s:
            res[i] += c
            if i == 0 or i == numRows - 1:
                flag = -flag
            i += flag
        return ''.join(res)
~~~



### 7.整数反转

### 8. 字符串转整数

~~~python
class Solution:
    def myAtoi(self, str: str) -> int:
        if not str:
            return 0
        while str and str[0] == ' ':  # 去除前面的空格
            str = str[1:]
        if not str:
            return 0
        
        if str[0] not in '0123456789+-':  # 判断合法性
            return 0
        
        ab = 1
        if str[0] == '-':  # 判断符号
            ab = -1
            str = str[1:]
        elif str[0] == '+':
            str = str[1:]
        else:
            str = str
            
        x = ''
        for i in str:
            if i in '0123456789':  # 获取数字
                x += i
            else:
                break
        if x:
            res = int(x)*ab  # 判断溢出
            if res > 2**31-1:
                return 2**31 - 1
            elif res < -2**31:
                return -2**31
            else:
                return res
        else:
            return 0
~~~



### 9. 回文数

* 将整数拆解，看是否回文

### 10. 正则表达式匹配

~~~python
class Solution:
    def isMAtch(self, text, pattern):
        if not pattern:
            return not text
        first_match = bool(text) and pattern[0] in {text[0],'.'}
        if len(pattern) > 1 and pattern[1] == '*':
            return self.isMatch(text,pattern[2:]) or first_match and self.isMatch(text[1:],pattern)
        return first_match and self.isMatch(text[1:],pattern[1:])
~~~



### 11. 盛水最多的容器

* 双指针法

~~~python
class Solution:
    def maxArea(self, height: List[int]) -> int:
        if len(height) < 2:
            return 0
        m = 0
        left = 0
        right = len(height) - 1
        while left < right:
            if height[left] < height[right]:
                m = max(height[left]*(right-left), m)
                left += 1
            else:
                m = max(height[right]*(right-left), m)
                right -= 1
        
        return m
~~~



### 12. 整数转罗马数字

* 贪心算法，分情况讨论

~~~python
class Solution:
    def intToRoman(self, num):
        nums = [int(i) for i in str(num)[::-1]]
        s_dic = {1:'I',2:'X',3:'C',4:'M'}
        c_dic = {1:'V',2:'L',3:'D'}
        
        rel = []
        for i,n in enumerate(nums):
            if n == 0:
                continue
            elif 0 < n < 4:
                rel.append(s_dic[i+1]*n)
            elif n == 4:
                rel.append(s_dic[i+1]+c_dic[i+1])
            elif 4 < n < 9:
                l = n%5
                rel.append(c_dic[i+1]+s_dic[i+1]*l)
            else:
                rel.append(s_dic[i+1]+s_dic[i+2])
        return ''.join(rel[::-1])
        
        
~~~



### 13. 罗马数字转整数

* 小的在左边做减法，在右边做加法

### 14. 最长公共前缀

* 水平扫描

~~~python
class Solution:
    def judge(self,a,b):
        if len(b) < len(a):
            a, b = b, a
        rel = ''
        for i,x in enumerate(a):
            if x != b[i]:
                return b[:i]
        return a
        
    def longestCommonPrefix(self,strs):
        if len(strs) < 1:
            return ''
        ans = strs[0]
        for s in strs[1:]:
            ans = self.judge(ans,s)
            if not ans:
                return ans
            
        return ans
~~~



### 15. 三数之和

* 先排序，再用两个指针寻找，跳过重复值

~~~python
class Solution:
    def threeSum(nums):
        nums.sort()
        ans = []
        for i,num in enumerate(nums[:-2]):
            if i==0 or nums[i] > nums[i-1]:
                l = i+1
                r = len(nums) - 1
                x = num + nums[l] + nums[r]
                if x == 0:
                    ans.append([num, nums[l], nums[r]])
                    l += 1
          			r -= 1
                    while l < r and nums[l] == nums[l-1]:
                        l += 1
                    while l < r and nums[r] == nums[r+1]:
                        r -= 1
                elif x < 0:
                    l += 1
                else:
                    r -= 1
       return ans
        
~~~



### 16. 最接近的三数之和

* 先排序，后用双指针

~~~python
class Solution:
    def threeSunClosest(self, nums, target):
        nums.sort()
        temp = float('inf')
        
        for i,num in enumerate(nums[:-2]):
            l = 0
            r = len(nums)-1
            while l < r:
                x = num + nums[l] + nums[r] - target
                if x == 0:
                    return target
                elif x < 0:
                    l += 1
                else:
                    r -= 1
                if abs(x) < temp:
                    temp = abs(x)
                    rel = x
        return rel+target
            
            
            
~~~



### 17. 电话号码的字母组合

* 回溯法

~~~python
class Solution:
    def letterCombinations(self, digits):
        if not digits:
            return []
        
        dic = {'2':'abc','3':'def','4':'ghi','5':'jkl','6':'mno','7':'pqrs','8':'tuv','9':'wxyz'}
        ans = []
        for num in digit:
            st = dic[num]
            if not ans:
                ans = [i for in st]
            else:
                ans = [i+j for i in ans for j in st]
                
        return ans
~~~



### 18. 四数之和

* 排序后双指针

### 19. 删除链表的倒数第N个节点

* 两个指针相差n，快的指针到达结尾时，慢的指针即为所求节点

~~~python
class Solution:
    def removeNthFromEnd(self, head, n):
        quick = head
        slow = head
        res = slow
        
        for i in range(n):
            quick = quick.next
        if not quick:
            return head.next
        while quick.next:
            quick = quick.next
            slow = slow.next
        slow.next = slow.next.next
        return res
~~~



### 20. 有效的括号

* 使用栈存储，配对成功弹出，最后栈为空则为有效

~~~python
class Solution:
    def isValid(self, s: str) -> bool:
        if not s:
            return True

        tmp = []
        dic = {'}':'{',']':'[',')':'('}

        for i in s:
            if i in dic and tmp and tmp[-1] == dic[i]:
                tmp.pop()
            elif i not in dic:
                tmp.append(i)
            else:
                return False

        if tmp:
            return False
        return True
~~~



### 21.合并两个有序链表

* 递归，每次返回更小的头节点
* 迭代，循环判断

~~~python
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        if not l1:
            return l2
        if not l2:
            return l1
        node = ListNode(0)
        rel = node
        while l1 and l2:
            if l1.val < l2.val:
                node.next = ListNode(l1.val)
                node = node.next
                l1 = l1.next
            else:
                node.next = ListNode(l2.val)
                node = node.next
                l2 = l2.next
        if l1 != None:
            node.next = l1
        if l2 != None:
            node.next = l2
        return rel.next
~~~



### 22. 括号生成

* 回溯法

~~~python
class Solution():
    def generateParenthesis(self, N):
        ans = []
        
        def backtrack(s='',left=0, right=0):
            if len(s)==2*N:
                ans.append(s)
                return
            if left < N:
                backtrack(s+'(',left+1,right)
            if right < left:
                backtrack(s+')',left,right+1)
        backtrack()
        return ans
~~~



### 23. 合并k个排序链表

* 遍历所有链表，将所有节点的值放入数组，排序，新建有序链表
* 逐一比较
* 逐一两两合并
* 分治合并

~~~python
class Solution:
    def mergeTwoLists(self, node1, node2):
        if not node1:
            return node2
        if not node2:
            return node1
        
        if node1.val < node2.val:
            head = node1
            head.next = self.mergeTwoLists(node1.next, node2)
        else:
            head = node2
            head.next = self.mergeTwoLists(node1, node2.next)
        return head
    
    def merge(self, lists, l, r):
        if l > r:
            return
        if l == r:
            return lists[l]
        mid = (l+r)//2
        l1 = self.merge(lists, l, mid)
        l2 = self.merge(lists, mid+1, r)
        return self.mergeTwoLists(l1, l2)
    
    def mergeKLists(self, lists):
        return self.merge(lists, 0, len(lists)-1)
~~~



### 24. 两两交换链表中的节点

* 递归
* 迭代，三指针

~~~python
class Solution:
    def swapPairs(self,head):
        start = ListNode(0)
        start.next = head
        pre = start
        
        while pre.next and pre.next.next:
            cur = pre.next
            pre.next = cur.next
            cur.next = cur.next.next
            pre.next.next = cur
            pre = cur
        return start.next
~~~



### 25. k个一组翻转链表

* 递归

~~~python
class Solution:
    def reverseKGroup(self,head,k):
        count = 0
        cur = head
        
        while cur and count != k:
            count += 1
            cur = cur.next
            
        if count == k:
            cur = self.reverseKGroup(cur,k)
            for i in range(k):
                tmp = cur.next
                head.next = cur
                cur = head
                head = tmp
        	head = cur
        return head 
~~~



### 26. 删除排序数组的重复项

* 两个指针，遇到重复跳过

~~~python
class Solution:
    def removeDuplicates(self, nums):
        i = 0
        for j in range(1,len(nums)):
            if nums[j] != nums[i]:
                i += 1
                nums[i] = nums[j]
        return i+1
~~~



### 27. 移除元素

* 同26，两个指针

~~~python
class Solutin:
    def removeElement(self, nums, val):
        i = 0
        for j in range(1, len(nums)):
            if nums[j] != val:
                nums[i] = nums[j]
                i += 1
        return i
~~~



### 28. 实现strStr()

* 滑动窗口
* 双指针

~~~python
class Solution:
    def strStr(self, haystack, needle):
        if not needle:
            reutrn 0
        if len(needle) > len(haystack):
            return -1
        
        m, n = len(haystack), len(needle)
        for i in range(m-n+1):  # m-n+1
            if haystack[i:i+n] == needle:
                return i
        return -1
~~~



### 29. 两数相除

### 30. 串联所有单词的字串

~~~python
class Solution:
    def findSubstring(self, s, words):
        if not s or not words:
            return []
        
        len_s = len(s)
        m = len(words)
        n = len(words[0])
        words_dic = {}
        for word in words:
            words_dic[word] = words_dic.get(word,0) + 1
        ans = []
        
        for i in range(n):
            count = 0
            left,right = i,i
            dic = {}
            while right + n <= len_s:
                w = s[right:right+n]
                right += n
                dic[w] = dic.get(w,0) + 1
                count += 1
                while dic[w] > words_dic.get(w,0):
                    left_w = s[left:left+n]
                    left += n
                    dic[left_w] -= 1
                    count -= 1
                if count == m:
                    ans.append(left)
           return ans
~~~



### 31. 下一个排列

* 先找出最大的索引 k 满足 nums[k] < nums[k+1]，如果不存在，就翻转整个数组；再找出另一个最大索引 l 满足 nums[l] > nums[k]；交换 nums[l] 和 nums[k]；最后翻转 nums[k+1:]

### 32. 最长的有效括号

* 动态规划

~~~python
class Solution:
    def longestValidParentheses(self, s):
        if not s:
            return 0
        dp = [0]*len(s)
        for i in range(1,len(s)):
            if s[i] == ')':
                pre = i - dp[i-1] - 1
                if pre >= 0 and s[pre] == '(':
                    dp[i] = dp[i-1] + 2
                if pre > 0:
                    dp[i] += dp[pre-1]
        return max(dp)
~~~



### 33. 搜索旋转排序数组

* 二分查找，判断target在左边还是右边

~~~python
class Solution:
    def search(self, nums, target):
        n = len(nums)
        l, r = 0, n-1
        
        while l <= r:
            mid = (l+r)//2
            if nums[mid] == target:
                return mid
            if nums[0] <= nums[mid]:
            	if nums[0] <= target < nums[mid]:
                    r = mid - 1
                else:
                    l = mid + 1
            else:
                if nums[mid] < target <= nums[n-1]:
                    l = mid + 1
                else:
                    r = mid - 1
        return -1
~~~



### 34. 在排序数组中查找元素的第一个和最后一个位置

* 用二分法，先找左边界，再找找右边界

~~~python
class Solution:
    def searchRange(self, nums, target):
        res = [-1, -1]
        if not nums:
            return res
        l, r = 0, len(nums)-1
        while l < r:
            mid = (l+r)//2
            if nums[mid] >= target:
                r = mid
            else:
                l = mid + 1
        if nums[l] != target:
            return res
        res[0] = l
        while l < r:
            mid = (l+r)//2
            if nums[mid] <= target:
                l = mid + 1
            else:
                r = mid
        res[1] = l - 1
        return res
~~~



### 35.搜索插入位置

* 二分查找

~~~~python
class Solution:
    def searchInsert(self, nums, target):
        if nums[0] >= target:
            return 0
        if nums[-1] < target:
            return len(nums)
        l, r = 0, len(nums)
        while l < r:
            mid = (l+r)//2
            if nums[mid] == target:
                return mid
            if nums[mid] > target:
                r = mid
            else:
                l = mid + 1
        return l
~~~~



### 36. 有效的数独

* 用三个字典存储，进行一次遍历

### 37. 解数独

* 回溯法

### 38. 外观数列

### 39. 组合总和(数字不可重复选取)

* 回溯算法+剪枝

~~~python
class Solution:
    def combinationSum(self,candidates,target):
        candidates.sort()
        n = len(candidates)
        ans = []
        def back(i,tmp_sum,tmp):
            if tmp_sum > target or i==n:
                return
            if tmp_sum == target:
                ans.append(tmp)
                return
            back(i,tmp_sum+candidates[i],tmp+[candidates[i]])
            back(i+1,tmp_sum,tmp)
        back(0,0,[])
        return ans
~~~

### 40. 组合总和 II

~~~python
class Solution:
    def combinationSum2(self, candidates,target):
        candidates.sort()
        n = len(candidates)
        ans = []
        def back(i,tmp_sum.tmp):
            if tmp_sum > target:
                return
            if tmp_sum == target:
                ans.append(tmp)
                return
            for j in range(i,n):
                if tmp_sum+candidates[j] > target:
                    break
                if j>i and candidates[j] == candidates[j-1]:
                    continue
                back(j+1,tmp_sum+candidates[j],tmp+[candidates[j]])
            back(0,0,[])
            return ans
~~~



### 42. 接雨水

* 双指针维护左右最大值，指针向中间移动的过程中进行计算

~~~python
class Solution:
    def trap(self, height):
        l, r = 0, len(height)-1
        l_max, r_max = 0, 0
        ans = 0
        
        while l < r:
            if height[l] < height[r]:
                if height[l] >= l_max:
                    l_max = height[l]
                else:
                    ans += l_max - height[l]
                l += 1
            else:
                if height[r] >= r_max:
                    r_max = height[r]
                else:
                    ans += r_max - height[r]
                r -= 1
        return ans
~~~

### 44. 通配符匹配

~~~python 
class Solution:
    def isMatch(self, s, p):
        m, n = len(s), len(p)
        dp = [[False]*(n+1) for i in range(m)]
        dp[0][0] = True
        for i in range(1,n+1):
            if p[i-1] == '*':
                dp[0][i] = True
            else:
                break
        for i in range(1,m+1):
            for j in range(1,n+1):
                if p[j-1] == '*':
                    dp[i][j] = dp[i-1][j]|dp[i][j-1]
                elif p[j-1] == '?' or s[j-1]==p[j-1]:
                    dp[i][j] = dp[i-1][j-1]
        return dp[m][n]
~~~



### 46. 全排列

* 回溯法

~~~python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        res = []
        def backtrack(nums,temp):
            if not nums:
                res.append(temp)
                return
            for i in range(len(nums)):
                backtrack(nums[:i]+nums[i+1:],temp+[nums[i]])
        
        backtrack(nums,[])
        return res
~~~

### 47. 全排列 II

~~~python
def class Solution:
    def permuteUnique(self,nums):
        nums.sort()
        ans = []
        
        def back(nums,tmp):
            if not nums:
                ans.append(tmp)
                return
            for i in range(len(nums)):
                if i>0 and nums[i]==nums[i-1]:
                    continue
                else:
                    back(nums[:i]+nums[i+1:],tmp+[nums[i]])
        back(nums,[])
        return ans
~~~



### 50. Pow(x,n)

* 快速幂，递归

~~~python
class Solution:
    def fun(self,x,n):
        if n==1:
            return x
        if n%2==0:
            return self.fun(x,n//2)**2
        else:
            return self.fun(x,n-1)*x

    def myPow(self, x: float, n: int) -> float:
        if n==0:
            return 1
        if n<0:
            x = 1/x
            n = -n
        return self.fun(x,n)
~~~

### 51. N皇后

~~~python
class Solution:
    def solveNQueens(self,n):
        ans = []
        def valid(row,col,track):
            if col in track:
                return False
            for k in range(row):
                if row+col=k+track[k] or row-col=k-track[k]:
                    return False
            return True
        
        def back(row,track):
            if row == n:
                ans.append(track)
                return 
            for col in range(n):
                if valid(row,col,track):
                    back(row+1,track+[col])
        back(0,[])
        return [['.'*i+'Q'+'.'*(n-i-1) for i in l] for l in ans]
~~~



### 53. 最大子序和

* 贪心，每一步都选择最佳方案
* 分治

![](https://pic.leetcode-cn.com/3aa2128a7ddcf1123454a6e5364792490c5edff62674f3cfd9c81cb7b5e8e522-file_1576478143567)

* 动态规划

~~~python
def Solution:
    def maxSubArray(self, nums):
        dp = [0]*len(nums)
        dp[0] = nums[0]
        ans = nums[0]
        for i in range(1,len(nums)):
            dp[i] = max(dp[i-1]+nums[i], nums[i])
        	ans = max(ans, dp[i])
        return ans
~~~



### 56. 合并区间

~~~python
class Solution:
    def merge(self, intervals):


        intervals.sort()
        rel = []
        for l in intervals:
            if not rel or l[0]>rel[-1][1]:
                rel.append(l)
            else:
                rel[-1][1] = max(rel[-1][1],l[1])
        return rel
~~~



### 58. 最后一个单词的长度

* 从后向前遍历，从第一个不是空格的字符开始

### 60. 第k个排列

~~~python
class Solution:
    def getPermutation(self,n,k):
        facts, nums=[1],['1']
        for i in range(1,n):
            facts.append(facts[i-1]*i)
            nums.append(str(i+1))
            
        k -= 1
        ans = []
        for i in range(n-1,-1,-1):
            idx = k//facts[i]
            k -= idx*facts[i]
            ans.append(nums[idx])
            del nums[idx]
            
        return ''.join(ans)
~~~



### 61. 旋转链表

* 先遍历一遍得到链表长度，再遍历第二遍进行旋转

~~~python
class Solution:
    def rotateRight(self, head, k):
        if not head:
            return None
        
        first = head
        second = head
        count = 1
        
        while first.next:
            first = first.next
            count += 1
        first.next = head
        
        for i in range(count - k%count - 1):
            second = second.next
            
        new_head = second.next
        second.next = None
        
        return new_head
~~~

### 62. 不同路径

~~~python 
class Solution:
    def uniquePaths(self, m, n):
        dp = [[0]*n for i in range(m)]
        for i in range(m):
            dp[i][0] = 1
        for j in range(n):
            dp[0][j] = 1
        for i in range(1,m):
            for j in range(1,n):
                dp[i][j] = dp[i-1][j] + dp[i][j-1]
        return dp[-1][-1]
        
~~~



### 64. 最小路径和

~~~python
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        m = len(grid)
        n = len(grid[0])
        dp = [[0]*n for i in range(m)]
        dp[0][0] = grid[0][0]

        for i in range(1,m):
            dp[i][0] = dp[i-1][0] + grid[i][0]
        for j in range(1,n):
            dp[0][j] = dp[0][j-1] + grid[0][j]

        for i in range(1,m):
            for j in range(1,n):
                dp[i][j] = min(dp[i-1][j],dp[i][j-1])+grid[i][j]
        return dp[-1][-1]
~~~



### 66. 加一

### 67. 二进制求和

* 逐位运算
* 位操作

### 69. x的平方根

* 二分查找

~~~python
class Solution:
    def mySqrt(self,x):
        l, r = 0, x
        ans = -1 
        while l <= r:
            mid = (l+r)//2
            y = mid*mid
            if y <= x:
                l = mid + 1
                ans = mid
            else:
                r = mid - 1
        return ans
~~~



### 70. 爬楼梯

* 斐波那契数列，动态规划

$$
dp[i] = dp[i-1] + dp[i-2]
$$

~~~python
class Solution:
    def climbStairs(self, n: int) -> int:
        if n == 1:
            return 1
        dp = [0]*n
        dp[0],dp[1] = 1,2
        for i in range(2,n):
            dp[i] = dp[i-2]+dp[i-1]
        return dp[-1]
~~~

### 71. 简化路径

~~~python
class Solution:
    def simplifyPath(self,path)
        tmp = []
        for i in path.split('/'):
            if i not in ['','.','..']:
                tmp.append(i)
            elif i == '..':
                tmp.pop()
                
        return '/'+'/'.join(tmp)
~~~



### 72. 编辑距离

~~~python
class Solution:
    def minDistanne(self,word1,word2):
        m = len(word1)
        n = len(word2)
        
        dp = [[0]*(n+1) for i in range(m+1)]
        for i in range(m+1):
            dp[i][0] = i
        for j in range(n+1):
            dp[0][j] = j
        for i in range(1,m+1):
            for j in range(1,n+1):
                if word1[i-1]==word2[j-1]:
                    v = 0
                else:
                    v = 1
                dp[i][j] = min(dp[i-1][j]+1,dp[i][j-1]+1,dp[i-1][j-1]+v)
        return dp[-1][-1]
~~~



### 74. 搜索二维矩阵

* 按一维矩阵二分法来计算

~~~python
class Solution:
    def searchMartrix(self, matrix, target):
        if not matrix:
            return False
        m = len(matrix)
        n = len(matrix[0])
        l, r = 0, m*n-1
        while l <= r:
            mid = (l+r)//2
            if matrix[mid//n][mid%n] == target:
                return True
            if matrix[mid//n][mid%n] > target:
                r = mid - 1
            else:
                l = mid + 1
        return False
~~~

### 77. 组合

~~~python
class Solution:
    def combine(self,n,k):
        ans = []
        def back(i,tmp):
            if len(tmp)==k:
                ans.append(tmp[:])
            for j in range(i,n+1):
                tmp.append(j)
                back(j+1,tmp)
                tmp.pop()
        back(1,[])
        return ans
    
~~~

### 78. 子集

~~~python
class Solution:
    def subsets(self,nums):
        ans = []
        
        def back(i,tmp,k):
            if len(tmp) == k:
                ans.append(tmp)
            for j in range(i,len(nums)):
                tmp.append(nums[j])
                back(i+1,tmp,k)
                tmp.pop()
                
        for k in range(len(nums)+1):
            back(0,[],k)
        return ans
            
~~~

### 79. 单词搜索

~~~python
class Solution:
    def exist(self,board,word):
        def dfs(i,j,k):
            if board[i][j] != word[k]:
                return False
            if k == len(word)-1:
                return True
            board[i][j] += ' '
            for p,q in [[i,j-1],[i,j+1],[i-1,j],[i+1,j]]:
                if 0<=p<m and 0<=q<n:
                    if dfs(p,q,k+1):
                        return True
            board[i][j] = board[i][j][0]
            return False
        
        m,n = len(board),len(board[0])
        for i in range(m):
            for j in range(n):
                if dfs(i,j,0):
                    return True
        return False
~~~

### 82. 删除排序链表中的重复元素

~~~python
class Solution:
    def deleteDuplicates(self,head):
        first = ListNode(0)
        first.next = head
        
        pre,cur = first,head
        while cur:
            while cur.next and cur.val==cur.next.val:
                cur = cur.next
            if pre.next == cur:
                pre = pre.next
            else:
                pre.next = cur.next
            
            cur = cur.next
        return first.next
~~~



### 83. 删除链表中的重复元素

~~~python
class Solution:
    def deleteDuplicates(self,head):
        if not head and not head.next:
            return head
        pre,cur = head,head.next
        while cur:
            if pre.val == cur.val:
                pre.next = cur.next
            else:
                pre = pre.next
            cur = cur.next
        return head
~~~

### 84. 柱状图中最大的矩形

用两个列表存储左右边界，用栈存储状态

~~~python
class Solution:
    def largestRectangleArea(self,heights):
        n = len(heights)
        left = [0]*n
        right = [n]*n
        stack = []
        
        for i in range(n):
            while stack and heights[stack[-1]]>=heights[i]:
                right[stack[-1]] = i
                stack.pop()
            if stack:
                left[i] = stack[-1]
            else:
                left[i] = -1
        if n!=0:
            ans = max([(right[i]-left[i]-1)*height[i] for i in range(n)])
        else:
            ans = 0
        return ans
~~~



### 85. 最大矩形

~~~python
class Solution:
    def maximalRectangle(self,matrix):
        if not matrix:
            return 0
        m = len(matrix)
        n = len(matrix[0])
        
        h = [0]*n
        l = [0]*n
        r = [n]*n
        ans = 0
        
        for i in range(m):
            cur_l,cur_r = 0,n
            for j in range(n):
                if matrix[i][j] == '1':
                    h[j] += 1
                else:
                    h[j] = 0
            for j in range(n):
                if matrix[i][j] == '1':
                    l[j] = max(l[j],cur_l)
                else:
                    l[j] = 0
                    cur_l = j+1
            for j in range(n-1,-1,-1):
                if matrix[i][j] == '1':
                    r[j] = min(r[j],cur_r)
                else:
                    r[j] = n
                    cur_r = j
            for j in range(n):
                if matrix[i][j] == '1':    
                    ans = max(ans,h[j]*(r[j]-l[j]))
        return ans
~~~

### 86. 分隔链表

~~~python
class Solution:
    def partition(self,head,x):
        if not head:
            return head
        l = ListNode(0)
        r = ListNode(0)
        node1 = l
        node2 = r
        
        while head:
            if head.val < x:
                node1.next = ListNode(head.val)
                node1.next = node1
            else:
                node2.next = ListNode(head.val)
                node2.next = node2
            head = head.next
        node1.next = r.next
        return l.next
~~~



### 88. 合并两个有序数组

* 合并后排序
* 双指针，从前往后，从后往前

### 92. 反转链表 II

~~~python
def reverse_node(head,m,n):
    pre,cur = None,head
    for i in range(m-1):
        pre = cur
        cur = cur.next
        
    pre_end = pre
    reverse_end = cur
    
    for i in range(n-m+1):
        nex = cur.next
        cur.next = pre
        pre = cur
        cur = nex
        
    if pre_end:
        pre_end.next = pre
    else:
        head = pre
    reverse_end.next = cur
    return head
~~~

### 93. 复原IP地址

~~~python
class Solution:
    def restoreIpAddresses(self,s):
        ans = []
        
        def back(s,tmp):
            if len(s)==0 and len(tmp)==4:
                ans.append('.'.join(tmp))
                return
            for i in range(min(3,len(s))):
                p,q = s[:i+1],s[i+1:]
                if p and 0<=int(p)<=255 and str(int(p))==p:
                    back(q,tmp+[p])
        back(s,[])
        return ans
~~~

### 94. 二叉树的中序遍历

~~~python
class Solution:
    def inorderTraversal(self,root):
        if not root:
            return []
        return self.inorderTraversal(root.left)+[root.val]+self.inorderTraversal(root.right)
~~~



### 95. 不同的二叉搜索树 II

~~~python
class Solution:
    def generateTrees(self, n):
        if n == 0:
            return []
        
        def fun(l,r):
            if l > r:
                return [None]
            
            trees = []
            for i in range(l,r+1):
                left = fun(l,i-1)
                right = fun(i+1,r)
                
                for li in left:
                    for ri in right:
                        cur = TreeNode(i)
                        cur.left = li
                        cur.right = ri
                        trees.append(cur)
            return trees
        return fun(1,n)
~~~

### 96. 不同的二叉搜索树

~~~python
class Solution:
    def numTrees(self,n):
        dp = [0]*(n+1)
        dp[0],dp[1] = 1,1
        
        for i in range(2,n+1):
            for j in range(1,i+1):
                dp[i] += dp[j-1]*dp[i-j]
        return dp[-1]
~~~



### 97. 交错字符串

~~~python
class Solution:
    def isInterleave(self,s1,s2,s3):
        m, n, t = len(s1),len(s2), len(s3)
        if m+n != t:
            return False
        dp = [[False]*(n+1) for i in range(m+1)]
        dp[0][0] = True
        for i in range(1, m+1):
            dp[i][0] = dp[i-1] & (s1[i-1]==s3[i-1])
        for j in range(1,n+1):
            dp[0][j] = dp[j-1] & (s2[j-1]==s3[j-1])
        for i in range(1,m+1):
            for j in range(n+1):
                dp[i][j] = (dp[i-1] & (s1[i-1]==s3[i-1])) or (dp[j-1] & (s2[j-1]==s3[j-1]))
~~~



### 98. 验证二叉搜索树

~~~python
class Solution:
    def isValidBST(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        def helper(node, lower = float('-inf'), upper = float('inf')):
            if not node:
                return True
            
            val = node.val
            if val <= lower or val >= upper:
                return False

            if not helper(node.right, val, upper):
                return False
            if not helper(node.left, lower, val):
                return False
            return True

        return helper(root)
~~~



### 100. 相同的树

* 递归

~~~python
class Solution:
    def isSameTree(self,p,q):
        if not p and not q:
            return True
        if not p or not q:
            return False
        
        if p.val==q.val and self.isSameTree(p.left,q.left) and self.isSameTree(p.right,q.right):
            return True
        return False
~~~



* 迭代

### 101.对称二叉树

* 递归，(t1.val == t2.val)&& isMirror(t1.right, t2.left)&& isMirror(t1.left, t2.right
* 迭代，用队列判断，连续两个点应该时相等的，两个节点的左右子节点按相反的方向插入队列

~~~python
def Solution:
    def isSymmetric(self,root):
        if not root:
            return True
        
        def fun(l,r):
            if not l and not r:
                return True
            if not l or not r:
                return False
            return l.val==r.val and fun(l.left,r.right) and fun(l.right,r.left)
        
        return fun(root.left,root.right)
~~~

### 102.宽度优先

~~~python
class Solution:
    def levelOrder(self,root):
		if not root:
            return []
        ans = []
        tmp = [root]
        
        while tmp:
            r = []
            for i in range(len(tmp)):
                cur = tmp.pop(0)
                r.append(cur.val)
                if cur.left:
                    tmp.apend(cur.left)
                if cur.right:
                    tmp.append(cur.right)
            ans.append(r)
        return ans
~~~

### 103. 二叉树的锯齿形层次遍历

~~~python
class Solution:
    def zigzagLevelOrder(self, root):
        if not root:
            return []
        ans = []
        tmp = [root]
        count = 0
        
        while tmp:
            r = []
            for i in range(len(tmp)):
                cur = tmp.pop(0)
                r.append(cur.val)
                if count%2 == 0:
                    if cur.left:
                        tmp.append(cur.left)
                    if cur.right:
                        tmp.append(cur.right)
                else:
                    if cur.right:
                        tmp.append(cur.right)
                    if cur.left:
                        tmp.append(cur.left)
            ans.append(r)
            tmp.reverse()
            count += 1
        return ans      
~~~



### 104. 二叉树的最大深度

* 递归（深度优先）

~~~python
class Solution:
    def maxDepth(self,root):
        if not root:
            return 0
        return max(self.maxDepth(root.left),self.maxDepth(root.right))+1
~~~



* 借用栈进行迭代

### 105. 从前序与中序遍历序列构造二叉树

~~~python
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        if not preorder:
            return None

        x = preorder[0]
        root = TreeNode(x)
        index = inorder.index(x)
        
        root.left = self.buildTree(preorder[1:index+1],inorder[:index])
        root.right = self.buildTree(preorder[index+1:],inorder[index+1:])
        return root
~~~

### 106. 从中序与后序遍历构造二叉树

~~~python
class Solution:
    def buildTree(self, inorder: List[int], postorder: List[int]) -> TreeNode:
        if not postorder:
            return None

        x = postorder[-1]
        root = TreeNode(x)
        index = inorder.index(x)

        root.left = self.buildTree(inorder[:index],postorder[:index])
        root.right = self.buildTree(inorder[index+1:],postorder[index:-1])
        return root
~~~



### 107. 二叉树的层次遍历 II

* 层序遍历后逆序

~~~python
class Solution:
    def levelOrderBottom(self,root):
        if not root:
            return []
        ans = []
        tmp = [root]
        
        while tmp:
            x = []
            a = []
            for r in tmp:
                x.append(r.val)
                if r.left:
                    a.append(r.left)
                if r.right:
                    a.append(r.right)
            ans.append(x)
            tmp = a
        return ans[::-1]              
~~~



### 108. 将有序数组转换为二叉搜索树

~~~python
class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> TreeNode:
        if not nums:
            return None
        n = len(nums)

        mid = (n-1)//2
        root = TreeNode(nums[mid])
        l = self.sortedArrayToBST(nums[:mid])
        root.left = l
        r = self.sortedArrayToBST(nums[mid+1:])
        root.right = r
        
        return root
~~~

### 109. 有序链表转换二叉搜索树

* 转为列表

~~~python
class Solution:
    def sortedListToBST(self, head: ListNode) -> TreeNode:
        lis = []
        while head:
            lis.append(head.val)
            head = head.next
        def fun(lis):
            if not lis:
                return None
            n = len(lis)
            mid = (n-1)//2
            root = TreeNode(lis[mid])
            root.left = fun(lis[:mid])
            root.right = fun(lis[mid+1:])
            return root
        return fun(lis)
~~~





### 110. 平衡二叉树

* 自顶向下的递归，如果有一边子树不存在，则应该等于另一边的深度

~~~python
class Solution:
    def isBalanced(self,root):
        def fun(root):
            if not root:
                return 0
            l = fun(root.left)
            if l == -1:
                return -1
            r = fun(root.right)
            if r == -1:
                return -1
            if abs(l-r)>1 :
                return -1
            return max(l,r)+1
        return fun(root)!=-1
~~~

### 111. 二叉树的最小深度

~~~python
class Solution:
    def minDepth(self,root):
        if not root:
            return 0
        l,r = 0,0
        if root.left:
            l = self.minDepth(root.left)
            
        if root.right:
            r = self.minDepth(root.right)
        if not root.left or not root.right:
            return l+r+1
        return min(l,r)+1
~~~



### 112. 路径总和

* 递归

### 113. 路径总和 II

* 给定二叉树和目标和，找到所有从根节点到叶子节点路径总和等于给定目标的路径



~~~python
# DFS
class Solution:
    def pathSum(self,root,sum):
        ans = []
        def dfs(root,sum,tmp):
            if not root:
                return
            sum -= root.val
            if not root.left and not root.right and sum==0:
                ans.append(tmp+[root.val])
            dfs(root.left,sum,tmp+[root.val])
            dfs(root.right,sum,tmp+[root.val])
        dfs(root,sum,[])
        return ans




################################## 
# BFS 

class Solution:
    def pathSum(self, root: TreeNode, sum_: int) -> List[List[int]]:
        if not root:
            return []
        
        stack = [(root, [root.val])]
        res = list()
        while stack:
            node, tmp = stack.pop(0)
            if not node.left and not node.right and sum(tmp) == sum_:
                res.append(tmp)
            if node.left:
                stack.append((node.left,tmp+[node.left.val]))
            if node.right:
                stack.append((node.right,tmp+[node.right.val]))
        
        return res
~~~



### 114. 二叉树的前序遍历

~~~python
class Solution:
    def preorderTraversal(self,root):
        if not root:
            return []
        return [root.val] + self.preorderTraversal(root.left)+self.preorderTraversal(root.right)
~~~



### 118. 杨辉三角

* 动态规划

### 119. 杨辉三角 II

* 公式 C(n,m) = n!/(n-m)!*m!
* 第i+1项是第i项的 (n-i)/(i+1) 倍

### 121. 买股票的最佳时机

* 遍历一遍数组，维护两个变量，截至当前的历史最低价和截至当前的最大利润

### 122. 买股票的最佳时机 II

* 贪心算法，if (prices[i] > prices[i - 1])   maxprofit += prices[i] - prices[i - 1]

### 124. 二叉树的最大路径和

* 递归法

~~~python
class Solution:
    def __init__(self):
        self.maxSum = float("-inf")

    def maxPathSum(self, root: TreeNode) -> int:
        def maxGain(node):
            if not node:
                return 0

            # 递归计算左右子节点的最大贡献值
            # 只有在最大贡献值大于 0 时，才会选取对应子节点
            leftGain = max(maxGain(node.left), 0)
            rightGain = max(maxGain(node.right), 0)
            
            # 节点的最大路径和取决于该节点的值与该节点的左右子节点的最大贡献值
            priceNewpath = node.val + leftGain + rightGain
            
            # 更新答案
            self.maxSum = max(self.maxSum, priceNewpath)
        
            # 返回节点的最大贡献值
            return node.val + max(leftGain, rightGain)
   
        maxGain(root)
        return self.maxSum
~~~



### 125. 验证回文串

### 127. 单词接龙



### 130.被围绕的区域

~~~python
class Solution:
    def solve(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """

        if not board: return
  
        row = len(board)
        col = len(board[0])
  
        if row < 3 or col < 3: return
        def dfs(i, j):
            if i < 0 or j < 0 or i >= row or j >= col or board[i][j] != 'O':
                return
            board[i][j] = '#'
            dfs(i - 1, j)
            dfs(i + 1, j)
            dfs(i, j - 1)
            dfs(i, j + 1)
        
        for i in range(row):
            dfs(i, 0)
            dfs(i, col - 1)
        
        for j in range(col):
            dfs(0, j)
            dfs(row - 1, j)

        for i in range(0, row):
            for j in range(0, col):
                if board[i][j] == 'O':
                    board[i][j] = 'X'
                if board[i][j] == '#':
                    board[i][j] = 'O'

~~~





### 131. 分割回文串

* 回溯

~~~python
class Solution(object):
    # 本题采用回溯法
    def partition(self, s):
        """
        :type s: str
        :rtype: List[List[str]]
        """
        # 定义一列表，用来保存最终结果
        split_result = []
        # 如果给定字符串s为空，则没有分割的必要了
        if len(s) == 0:
            return split_result

        def back(start=0, res=[]):
            if start >= len(s):
                split_result.append(res)
                return 
            for end in range(start+1, len(s)+1):
                split_s = s[start:end]
                # 如果当前子串为回文串，则可以继续递归
                if split_s == s[start:end][::-1]:
                    back(end, res+[split_s])

        back()
        return split_result
~~~



### 136. 只出现一次的数字

* 列表
* 哈希表
* 2 * sum(set(nums)) - sum(nums)
* 位操作

### 138. 复制带随机指针的链表

* 

### 141. 环形链表

* 双指针

~~~python
class Solution:
    def hasCycle(self, head: ListNode) -> bool:
        if not head:
            return False
        if not head.next:
            return False

        fast = head.next
        slow = head

        while slow != fast:
            if not fast or not fast.next:
                return False
            fast = fast.next.next
            slow = slow.next
        return True
~~~

### 142. 环形链表 II

* 先找到相遇节点，再找到入口

~~~python
class Solution:
    def detectCycle(self, head: ListNode) -> ListNode:
        if not head:
            return None
        if not head.next:
            return None
        
        fast = head.next.next
        slow = head.next
        s = 1

        while fast != slow:
            if not fast or not fast.next:
                return None
            fast = fast.next.next
            slow = slow.next
            s += 1
        
        ss = head

        while fast != ss:
            fast = fast.next
            ss = ss.next

        return ss
~~~

### 145. 二叉树的后序遍历

~~~python
class Solution:
    def postorderTraversal(self,root):
        if not root:
            return []
        tmp = [root]
        ans = []
        while tmp:
            node = tmp.pop()
            ans.append(node.val)
            if node.left:
                tmp.append(node.left)
            if node.right:
                tmp.append(node.right)
                
        return ans[::-1]
~~~



### 148. 排序链表



### 151. 翻转字符串里的单词

~~~python
class Solution:
    def reverseWords(self,s):
        left,right = 0,len(s)-1
        while left<=right and s[left]==' ':
            left += 1
        while left<=right and s[right]==' ':
            right -= 1
        d = []
        word = ''
        while left<=right:
            if s[left]==' ' and word:
                d = [word] + d
                word = ''
            elif s[left]!=' ':
                word += s[left]
            left += 1
        d = [word]+d
        
        return ' '.join(d)
~~~





### 153. 寻找旋转排序数组中的最小值

~~~python
class Solution:
    def findMin(self, nums):
        n = len(nums)
        l, r = 0, n-1
        
        while l < r:
            mid = (l+r)//2
            if nums[mid] < nums[r]:
                r = mid
            else:
                l = mid + 1
        return nums[l]
~~~

### 154. 寻找旋转排序数组中的最小值 II

~~~python
class Solution:
    def findMin(self, nums):
        n = len(nums)
        l, r = 0, n-1
        
        while l < r:
            mid = (l+r)//2
            if nums[mid] < nums[r]:
                r = mid
            elif nums[mid] > nums[r]:
                l = mid + 1
            else:
                r -= 1
        return nums[l]
~~~



### 155. 最小栈

* 辅助栈和数据栈同步

### 160.相交链表

* 哈希表法：把A中每个结点的地址和应用存储在哈希表中
* 双指针，pA到达链表的尾部时，将它重新定位到B的头节点，pB到达链表的尾部时，将它重新定位到A的头节点，若在某个时刻pA和pB相遇，则为相交结点

### 162. 寻找峰值

~~~python
class Solution:
    def findPeakElement(self, nums):
        l, r = 0, len(nums)-1
        while l < r:
            mid = (l+r)//2
            if nums[mid] > nums[mid+1]:
                r = mid
            else:
                l = mid + 1  # 右边比mid高，所以mid一定不是峰值了
        return l
~~~



### 167. 两数之和 II - 输入有序数组

* 双指针

### 169. 多数元素

* 哈希表
* 排序，返回中间那个数

### 171. Excel表列序号

* 26进制

### 189. 旋转数组

### 191. 位1的个数

* 循环
* 位移动

### 198.打家劫舍

* 动态规划，

$$
f(k) = max(f(k-2)+A_k,f(k-1))
$$

### 199. 二叉树的右视图

~~~python
class Solution:
    def rightSideView(self,root):
        if not root:
            return []
        q = [root]
        ans = []
        
        while q:
            ans.append(q[-1].val)
            tmp = []
            for i in range(len(q)):
                node = q.pop(0)
                if node.left:
                    tmp.append(node.left)
                if node.right:
                    tmp.append(node.right)
            q = tmp
        return ans
~~~



### 200. 岛屿数量

~~~python
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        if not grid:
            return 0
        
        m = len(grid)
        n = len(grid[0])

        def dfs(i,j):
            if i<0 or j<0 or i>=m or j >= n or grid[i][j] != '1':
                return 
            grid[i][j] = '0'
            dfs(i-1,j)
            dfs(i,j-1)
            dfs(i+1,j)
            dfs(i,j+1)


        rel = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == '1':
                    rel += 1
                    dfs(i,j)
        return rel
~~~



### 203. 移除链表元素

* 哨兵节点-伪头

### 205. 同构字符串

* 哈希表

### 206. 反转链表

* 迭代，三个指针
* 递归

~~~python
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        if not head or not head.next:
            return head

        pre,cur = None,head
        while cur:
            nex = cur.next
            cur.next = pre
            pre = cur
            cur = nex

        return pre
~~~

### 215. 数组中的第k个最大元素

~~~python
import heapq
class Solution:
    def findKthLargest(self, nums, k):
        res = nums[:k]
        heapq.heapify(res)
        for num in nums[k:]:
            if num > res[0]:
                heapq.heappop(res)
            	heapq.heappush(res, num)
        return res[0]
~~~



### 217. 存在重复元素

### 219. 存在重复元素 II

* 用散列表来维护k大小的滑动窗口

### 226. 翻转二叉树

* 递归

~~~python
class Solution:
    def invertTree(self,root):
        if not root:
            return None
        root.left,root.right = root.right,root.left
        self.invertTree(root.left)
        self.invertTree(root.right)
        return root
~~~

* 迭代

~~~python
class Solution:
    def invertTree(self.root):
        if not root:
            return None
        q = [root]
        while queue:
            tmp = q.pop(0)
            tmp.left,tmp.right = tmp.right,tmp.left
            if tmp.left:
                q.append(tmp.left)
            if tmp.right:
                q.append(tmp.right)
        return root
~~~



### 230. 二叉搜索树中第K小的元素

* 二叉搜索树的中序遍历是一个递增数列

~~~python
class Solution:
    def kthSmallest(self,root,k):
        def inorder(r):
            if not r:
                return []
            return inorder(r.left)+[r.val]+inorder(r.right)
        return inorder(root)[k-1]
    

# 迭代
class Solution:
    def kthSmallest(self, root, k):
        """
        :type root: TreeNode
        :type k: int
        :rtype: int
        """
        stack = []
        
        while True:
            while root:
                stack.append(root)
                root = root.left
            root = stack.pop()
            k -= 1
            if not k:
                return root.val
            root = root.right
~~~



### 231. 2的幂

* 位运算

###  234. 回文链表

* 复制到数组后用双指针
* 递归
* 定义快慢指针，快的到达末端时，慢的到达中间，反转后半部分进行比较

### 235. 二叉搜索树的最近公共祖先

* 根据p、q的值判断所处位置

~~~python
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if p.val < root.val and q.val < root.val:
            return self.lowestCommonAncestor(root.left, p, q)
        if p.val > root.val and q.val > root.val:
            return self.lowestCommonAncestor(root.right, p, q)

        return root
~~~

### 236. 二叉树的公共祖先

* 如果root就是p或q，则直接返回；用递归查找p和q的位置，如果p和q在同一边，则继续查找，否则返回root

~~~python
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if root in (None,p,q):
            return root

        L = self.lowestCommonAncestor(root.left,p,q)
        R = self.lowestCommonAncestor(root.right,p,q)
        if L == None:
            return R
        if R == None:
            return L
        return root
~~~



### 237. 删除链表中的节点

* 把下一个节点复制到当前节点

### 240. 搜索二位矩阵 II

~~~python
class Solution:
    def searchMatrix(self, matrix, target):
        if not matrix:
            return False
        m, n = len(matrix), len(matrix[0])
        i, j = 0, n-1
        while i < m and j >= 0:
            if matrix[i][j] == target:
                return True
            if matrix[i][j] > target:
                j -= 1
            else:
                i += 1
        return False
~~~



### 257. 二叉树的所有路径

* DFS，深度优先遍历，递归

### 263. 丑数

### 283. 移动零

* 第一次遍历时，j指针记录非0个数，将非0的数赋值给nums[j]，第二次遍历法末尾的元素都赋值为0
* 使用两个指针i和j，只要nums[i]!=0，就交换nums[i]和nums[j]

### 299. 猜数字游戏

### 300. 最长上升子序列

* 动态规划

~~~python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        
        if not nums:
            return 0
        dp = [1]*len(nums)

        for i in range(1,len(nums)):
            for j in range(i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[i],dp[j]+1)

        return max(dp)
~~~

### 316. 去除重复字母

~~~python
def Solution:
    def removeDuplicateLetters(s):
        stack = []
        seen = set()
        last_occurrence = {c:i for i ,c in enumerate(s)}
        
        for i,c in enumerate(s):
            if c not in seen:
                while stack and c<stack[-1] and i<last_occurrence[stack[-1]]:
                    seen.discard(stack.pop())
                seen.add(c)
                stack.append(c)
        return ''.join(stack)
~~~

### 322. 零钱兑换

* 时间复杂度O(Sn)，对于每个状态，要枚举n个面额来转移状态

~~~python
class Solution:
    def coinChange(self,coins,amount):
        dp = [float('inf')]*(amount+1)
        dp[0] = 0
        for coin in coins:
            for x in range(coin,amount+1):
                dp[x] = min(dp[x],dp[x-coin]+1)
        if dp[-1] == float('inf'):
            return -1
        return dp[-1]
~~~



### 331. 验证二叉树的前序序列化

~~~python
class Solution:
    def isValidSerialization(self,preorder):
        nodes = preorder.split(',')
        degree = 1
        for node in nodes:
            if degree == 0:
                return False
            if node == '#':
                degree -= 1
            else:
                degree += 1
        return degree == 0
~~~



### 345. 反转字符串中的元音字母

* 双指针

### 349. 两个数组的交集

### 367. 有效的完全平方数

* 二分查找
* 牛顿迭代，x = 1/2(x+num/x)

### 371. 两整数之和

* 位运算

### 383. 赎金信

### 387. 字符串中的第一个唯一字符

* 用字典存储字符数

### 389. 找不同

### 394. 字符串编码

~~~python
class Solution:
    def decodeString(self,s):
        stack = []
        num = 0
        res = ''
        for c in s:
            if c.isdigit():
                num = num*10+int(c)
            elif c == '[':
                stack.append((res,num))
                res,num = '',0
            elif c == ']':
                top = stack.pop()
                res = top[0]+res*top[1]
            else:
                res += c
        return res
~~~



### 401. 二进制手表

### 402. 移掉K位数字

~~~python
class Solution:
    def removeKdigits(self,num,k):
        if k==len(num):
            return '0'
        tmp = []
        count = 0
        for i in range(len(num)):
            while tmp and tmp[-1]>num[i] and count<k:
                tmp.pop()
                count += 1
            tmp.append(num[i])
        tmp = tmp[:len(num)-k]
        ans = ''.join(tmp)
        return str(int(ans))
~~~



### 405. 数字转换为十六进制数

### 409. 最长回文串

~~~python
class Solution:
    def longestPalindrome(self, s: str) -> int:
        s = list(s)
        temp = [s[0]]
        rel = 0
        for i in s[1:]:
            if i in temp:
                rel += 2
                temp.remove(i)
            else:
                temp.append(i)
        if temp != []:
            rel += 1
        return rel
~~~



### 415. 字符串相加

### 437. 路径总和 III

* dfs和递归

### 441. 排列硬币

### 443. 压缩字符串

* 双指针

### 447. 回旋镖的数量

### 448. 找到所有数组中消失的数字

* 将所有数作为数组下标，置对应数为负数，仍为正数的位置即为未出现过的数

### 453. 最小移动次数使数组相等

* 增加n-1个数相当于减少1个数，减少所有数到与最小值相等即可

### 461. 汉明距离

* 内置位计数功能
* 移位计数

### 463. 岛屿的周长

### 475. 供暖器

* 对于每个房屋，取最小距离，然后在这些最小距离中取最大值

### 485. 最大连续1的个数

### 501. 二叉搜索树中的众数

### 507. 完美数

### 509. 斐波那契数列

### 516. 最长回文子序列

~~~python
class Solution:
    def longestPalindromeSubseq(self, s: str) -> int:
        n = len(s)
        dp = [[0]*n for i in range(n)]
        for i in range(n):
            dp[i][i] = 1
        for i in range(n-1,-1,-1):
            for j in range(i+1,n):
                if s[i] == s[j]:
                    dp[i][j] = dp[i+1][j-1] + 2
                else:
                    dp[i][j] = max(dp[i+1][j],dp[i][j-1])
        return dp[0][n-1]
~~~



### 521. 最长特殊序列 I

### 538. 把二叉搜索树转换为累加树

* 回溯法，判断当前节点是否存在，存在几句递归右子树，然后更新总和和当前节点值，然后递归左子树
* 使用栈迭代

### 541. 反转字符串 II

### 542. 01矩阵

~~~python
class Solution:
    def updateMatrix(self, matrix: List[List[int]]) -> List[List[int]]:
        m, n = len(matrix), len(matrix[0])
        dp = [[0] * n for _ in range(m)]
        q = [(i, j) for i in range(m) for j in range(n) if matrix[i][j] == 0]
        # 将所有的 0 添加进初始队列中    
        seen = set(q)

        # 广度优先搜索
        while q:
            i, j = q.pop(0)
            for ni, nj in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]:
                if 0 <= ni < m and 0 <= nj < n and (ni, nj) not in seen:
                    dp[ni][nj] = dp[i][j] + 1
                    q.append((ni, nj))
                    seen.add((ni, nj))
        
        return dp
~~~



### 543.二叉树的直径

* 深度优先搜索，定义递归函数depth计算子树深度，子树深度为max(L, R) + 1，最大节点数为 L + R + 1

### 551. 学生出勤记录 I

### 557. 反转字符串中的单词 III

### 559. N叉树的最大深度

* 深度优先+递归

### 561. 数组拆分 I

### 563. 二叉树的坡度

* 递归

### 575. 分糖果

### 581. 最短无序连续子数组

* 排序后比较
* 使用栈
* 不使用额外空间，无序数组中最小元素的正确位置可以决定左边界，最大元素的正确位置可以决定右边界

### 589. N叉树的前序遍历

### 599. 两个列表的最小索引总和

* 哈希表

### 605. 种花问题

### 617. 合并二叉树

* 递归，对两棵树进行前序遍历，将对应的节点进行合并
* 迭代，利用栈

### 633. 平方数之和

* 双指针
* 二分查找
* 费马平方和定理
  * 一个非负整数 c 能够表示为两个整数的平方和，当且仅当 c 的所有形如 4k+3 的质因子的幂次均为偶数。

### 637. 二叉树的层平均值

* 

### 647. 回文子串

* 从2N+1个中心往两侧延申

```python
class Solution(object):
    def countSubstrings(self, S):
        N = len(S)
        ans = 0
        for center in xrange(2*N - 1):
            left = center / 2
            right = left + center % 2
            while left >= 0 and right < N and S[left] == S[right]:
                ans += 1
                left -= 1
                right += 1
        return ans
```

* 马拉车算法

### 680. 验证回文字符串 II

* 判断两次

```python
class Solution:
    def validPalindrome(self, s: str) -> bool:
        #双指针
        def helper(s1:str):
            l=0
            r=len(s1)-1
            while l<=r:
                if s1[l]!=s1[r]:
                    return l,r
                l+=1
                r-=1
                if l>r:
                    return l,r
        #第一次判别 
        l1,r1=helper(s)
        if l1>r1:
            return True
        # 当出现有不相等的字符，返回不相等的字符的区间的下标 ，左边部分[a,b]
        l,r=helper(s[l1:r1])
        if l>r:
            return True
        # 右边部分 比如[a,b,c] 放进判别的是[b,c]
        l,r=helper(s[l1+1:r1+1])
        if l>r:
            return True
        return False
```

### 695. 岛屿的最大面积

~~~python
class Solution:
    def dfs(self,grid,i,j):
        if cur_i<0 or cur_j<0 or cur_i==len(grid) or cur_j==len(grid[0]) or grid[cur_i][cur_j]!=1:
            return 0
        grid[cur_i][cur_j] = 0
        ans = 1
        for di,dj in [[0,1],[0,-1],[1,0],[-1,0]]:
            next_i,next_j = cur_i+di,cur_j+dj
            ans += self.dfs(grid,next_i,next_j)
        return ans
    
    def maxAreaOfIsland(self,grid):
        ans = 0
        for i,l in enumerate(grid):
            for j,n in enumerate(l):
                ans = max(self.dfs(grid,i,j),ans)
        return ans
    
    
class Solution:
    def maxAreaOfIsland(self,grid):
        ans = 0
        for i,l in enumerate(grid):
            for j,n in enumerate(l):
                cur = 0
                q = [[i,j]]
                while q:
                    [cur_i,cur_j] = q[0]
                    q.pop(0)
                    if cur_i<0 or cur_j<0 or cur_i==len(grid) or cur_j==len(grid[0]):
                        continue
                    cur += 1
                    grid[cur_i][cur_j]=0
                    for di,dj in [[0,1],[0,-1],[1,0],[-1,0]]:
                        next_i,next_j = cur_i+di,cur_j+dj
                        q.append([nex_i,next_j])
                ans = max(ans,cur)
~~~



### 704. 二分查找

~~~python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        l = 0
        r = len(nums)-1
        
        while l<=r:
            mid = (l+r)//2
            if nums[mid] == target:
                return mid
            if nums[mid] > target:
                r = mid - 1
            if nums[mid] < target:
                l = mid+1
        return -1
~~~

### 876. 链表的中间节点

* 快慢指针

~~~python
class Solution:
    def middleNode(self, head: ListNode) -> ListNode:
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        return slow

~~~



### 866. 回文素数

### 917. 仅仅反转字母

~~~python
class Solution:
    def reverseOnlyLetters(self, S: str) -> str:
        s =list(S)
        i = 0
        j = len(s)-1
        while i<j:
            if s[i].isalpha() and s[j].isalpha():
                s[i],s[j] = s[j],s[i]
                i += 1
                j -= 1
            if not s[i].isalpha():
                i += 1
            if not s[j].isalpha():
                j -= 1
        return ''.join(s)
~~~

### 946. 验证栈序列

~~~python
class Solution:
    def validateStacjSequences(self,pushed,popped):
        j = 0
        stack = []
        
        for x in pushed:
            stack.append(x)
            while stack and j<len(popped) and stack[-1]==popped[j]:
                stack.pop()
                j+=1
        return j==len(popped)
~~~



### 1095. 山脉数组中查找目标值

~~~python
class Solution:
    def binary_search(arr,target,l,r):
        while l<=r:
            mid = (l+r)//2
            if arr[mid] == target:
                return mid
            if arr[mid] < target:
                l = mid+1
            else:
                r = mid-1
        return -1
    
    def findInMountainArray(self,target,mountain_arr):
        l,r = 0,len(mountain_arr)-1
        while l<r:
            mid = (l+r)//2
            if mountain_arr[mid] < mountain_arr[mid+1]:
                l = mid+1
            else:
                r = mid
        peak = l
        
        index = binary_search(mointain_arr,target,0,peak)
        if index!=-1:
            return index
        mountain_arr = [-i for i in mountain_arr]
        index = binary_search(mountain_arr,-target,peak+1,len(mountain_arr)-1)
        return index
~~~



### 1143. 最长公共子序列

~~~python
def LCS(a,b):
    m,n = len(a),len(b)
    dp = [[0]*(n+1) for i in range(m+1)]
    for i in range(1,m+1):
        for j in range(1,n+1):
            if a[i-1] == b[i-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j],dp[i][j-1])
    return dp[-1][-1]
~~~

### 最长公共子串

~~~python
def fun(a,b):
    m,n = len(a),len(b)
    dp = [[0]*(n+1) for i in range(m+1)]
    ans = 0
    for i in range(1,m+1):
        for j in range(1,n+1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
                ans = max(ans,dp[i][j])
    return ans
~~~



### 1190. 反转每对括号间的字串

* 用栈保存左括号，遇到右括号时弹出

~~~python
class Solution:
    def reverseParentheses(self, s: str) -> str:
        stack = []
        temp = []
        for i,x in enumerate(s):
            if x == '(':
                stack.append(i)
            elif x == ')':
                left = stack.pop() + 1
                right = i
                temp.append([left,right])
        s = list(s)
        for num in temp:
            s[num[0]:num[1]] = s[num[0]:num[1]][::-1]
        s = [i for i in s if i not in '()']
        return ''.join(s)
~~~

### 1288. 删除被覆盖的区间

* 合并区间

~~~python
class Solution:
    def removeCoveredIntervals(self, intervals: List[List[int]]) -> int:
        if not intervals:
            return 0
        intervals.sort()
        ans = [intervals[0]]

        for i in intervals[1:]:
            if ans[-1][0] == i[0] and ans[-1][1] < i[1]:
                ans[-1] = i
            elif ans[-1][0] <= i[0] and ans[-1][1] >= i[1]:
                continue
            else:
                ans.append(i)

        return len(ans)
~~~





### 1328. 破坏回文串

* 分情况，前一半有非’a'，直接替换，没有则替换最后一个a

### 1332. 删除回文子序列

### 1339. 分裂二叉树的最大乘积

* 遍历求各子树和，最后分别计算其与剩下的树和的乘积，取最大值

~~~python
class Solution:
    def maxProduct(self,root):
        list_sum = []
        def dfs(node):
            if not node:
                return 0
            s = dfs(node.left)+dfs(node.right)+node.val
            list_sum.append(s)
            return s
        
        total = dfs(root)
        ans = float('-inf')
        for s in list_sum:
            ans = max(ans,s*(total-s))
        return ans%(10**9+7)
~~~

### 1400. 构造K个回文字符串

~~~python
class Solution:
    def canConstruct(self, s: str, k: int) -> bool:
        # 右边界为字符串的长度
        right = len(s)
        # 统计每个字符出现的次数
        occ = collections.Counter(s)
        # 左边界为出现奇数次字符的个数
        left = sum(1 for _, v in occ.items() if v % 2 == 1)
        # 注意没有出现奇数次的字符的特殊情况
        left = max(left, 1)
        return left <= k <= right
~~~

### 快手 距离N最近的斐波那契数

~~~python
def close_fib(n):
    fb0 = 0
    fb1 = 1
    while True:
        if n<fb1:
            left = n-fb0
            right = fb1-n
            if left<right:
                return fb0
            else:
                return fb1
        tmp = fb0+fb1
        fb0,fb1 = fb1,tmp
~~~

### 腾讯 找出数组中比左边大比右边的小的元素

* 用一个数组记录扫描到当前位置时,该元素前面的最大元素,再用一个数组记录,扫描到当前位置时,该元素后面的最小元素.最后用,当前位置的元素和扫描到当前位置时该元素前面的最大元素值,扫描到当前位置时该元素后面的最小元素值和它进行比较就可以得到一个boolean类型的数组，用来记录当前位置上的元素是否满足条件

~~~python
def fun(arr):
    max_arr = [i for i in arr]
    min_arr = [i for i in arr]
    
    for i in range(1,len(arr)):
        if max_arr[i] < max_arr[i-1]:
            max_arr[i] = max_arr[i-1]
    for i in range(len(arr)-2,-1,-1):
        if min_arr[i] > min_arr[i+1]:
            min_arr[i] = min_arr[i+1]
            
    ans = []
    for i in range(len(arr)):
        if arr[i]>=max_arr[i] and arr[i]<=min_arr[i]:
            ans.append(arr[i])
    return ans
~~~



