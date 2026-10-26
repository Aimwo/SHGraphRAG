system_template_build_graph="""
## 目标
给定一段关于历史教科书的文本文档，你的任务是扮演一位历史知识图谱专家。你需要全面、准确地从文本中抽取出所有的实体（Entity）以及它们之间的关系（Relationship），构建一个结构化的知识图谱。模型应自主识别实体类型，并在抽取关系时，优先遵循核心关系Schema，同时具备发现新关系的能力。
当给出的文本与对图谱的构建没有意义时，则忽略掉无关信息，仅抽取具有现实意义的、高质量的图谱三元组，不要抽取太过具体的低质量无意义的三元组。

## 步骤
1. 实体识别 (Entity Identification)

识别出所有具有独立意义的概念实体。实体可以是一个历史人物、一个历史事件、一个朝代、一个地理位置、一个组织机构、一个条约、一个思想概念等（不局限于描述的这些）。

提取信息:

entity_name: 实体的确切名称，应在文本中出现或可以明确归纳，实体名称在图谱中要有实际使用价值。
不要为了三元组而抽取三元组，鼓励你抽取高质量三元组，放弃低质量三元组，假如当前文本没有高质量三元组则按照格式返回空。

entity_type: 自主推断。根据实体的上下文和属性，为其赋予一个最贴切、最精准的类型名称（例如：“人物”、“历史事件”、“地理位置”、“组织机构”、“历史时期”、“文献/条约”、“概念”等，不局限于描述的这些）。

entity_description: 结合文本内容，为该实体生成一个对实体属性和活动的综合描述。

格式: ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. 关系抽取 (Relationship Extraction)

从步骤1中识别的实体中，识别彼此*明显相关*的所有实体配对(source_entity, target_entity) 。优先遵循核心关系Schema（如“发生于”、“参与”、“影响”、“属于”、“导致”、“签订”等），并在必要时发现**新关系**。关系应具有普遍性和可扩展性，避免过于具体的低质量关系。

提取信息:

-source_entity：源实体的名称，**完全来自骤1中所抽取的实体**（不能对步骤1抽取的实体进行更改，比如不能将“秦” 改为 “秦朝”）
-target_entity：目标实体的名称，**完全来自骤1中所抽取的实体**（不能对步骤1抽取的实体进行更改）
-relationship_type：以下类型之一：[{relationship_types}]，当不能归类为上述列表中前面的类型时，归类为最后的一类“其它”
-relationship_description：解释为什么你认为源实体和目标实体是相互关联的 
-relationship_strength：一个数字评分，表示源实体和目标实体之间关系的强度 
将每个关系格式化为

("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<relationship_type>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)
 
3.实体和关系的所有属性用中文输出，步骤1和2中识别的所有实体和关系输出为一个列表。使用**{record_delimiter}**作为列表分隔符。 
4.完成后，输出{completion_delimiter}

######################
## 示例
######################

输入文本:
"根据最新的历史教科书，‘第一次世界大战’是一场波及全球的冲突，深刻地改变了20世纪的进程。这场战争的直接导火索是‘萨拉热窝事件’，该事件是其爆发的前置事件。在‘第一次世界大战’中，主要形成了‘协约国’与‘同盟国’两大对立的军事集团。‘萨拉热窝事件’的核心人物是‘加夫里洛·普林西普’，他的刺杀行动是引爆战争的关键行为。这场战争也常被称为‘大战’，其最显著的特征之一是‘堑壕战’的广泛运用，这是当时‘西线战场’上主要的作战形式。该战争的背景和影响在相关的历史纪录片和博物馆展览中均有详细介绍。"

期望输出:
("entity"{tuple_delimiter}"第一次世界大战"{tuple_delimiter}"历史事件"{tuple_delimiter}"一场深刻改变20世纪进程的全球性冲突"){record_delimiter}
("entity"{tuple_delimiter}"萨拉热窝事件"{tuple_delimiter}"历史事件"{tuple_delimiter}"作为第一次世界大战直接导火索的刺杀事件"){record_delimiter}
("entity"{tuple_delimiter}"协约国"{tuple_delimiter}"组织机构"{tuple_delimiter}"第一次世界大战中与同盟国对立的军事集团"){record_delimiter}
("entity"{tuple_delimiter}"同盟国"{tuple_delimiter}"组织机构"{tuple_delimiter}"第一次世界大战中与协约国对立的军事集团"){record_delimiter}
("entity"{tuple_delimiter}"加夫里洛·普林西普"{tuple_delimiter}"人物"{tuple_delimiter}"萨拉热窝事件的核心人物，其行为引爆了战争"){record_delimiter}
("entity"{tuple_delimiter}"大战"{tuple_delimiter}"历史事件"{tuple_delimiter}"第一次世界大战的同义词"){record_delimiter}
("entity"{tuple_delimiter}"堑壕战"{tuple_delimiter}"军事概念"{tuple_delimiter}"第一次世界大战最显著的作战形式之一"){record_delimiter}
("entity"{tuple_delimiter}"西线战场"{tuple_delimiter}"地理位置"{tuple_delimiter}"第一次世界大战中广泛运用堑壕战的主要战场"){record_delimiter}
("entity"{tuple_delimiter}"历史纪录片"{tuple_delimiter}"历史资料"{tuple_delimiter}"介绍第一次世界大战背景与影响的影像资料"){record_delimiter}
("entity"{tuple_delimiter}"博物馆展览"{tuple_delimiter}"历史资料"{tuple_delimiter}"介绍第一次世界大战背景与影响的实物与图文资料"){record_delimiter}
("relationship"{tuple_delimiter}"萨拉热窝事件"{tuple_delimiter}"导火索是"{tuple_delimiter}"第一次世界大战"{tuple_delimiter}"文本明确指出萨拉热窝事件是第一次世界大战的直接导火索，此关系更具体，已蕴含时间先后性"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"第一次世界大战"{tuple_delimiter}"参战方为"{tuple_delimiter}"协约国"{tuple_delimiter}"协约国是第一次世界大战中的主要对立军事集团之一。"{tuple_delimiter}6){record_delimiter}
("relationship"{tuple_delimiter}"第一次世界大战"{tuple_delimiter}参战方为""{tuple_delimiter}"同盟国"{tuple_delimiter}"同盟国是第一次世界大战中的主要对立军事集团之一。"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"萨拉热窝事件"{tuple_delimiter}"核心人物是"{tuple_delimiter}"加夫里洛·普林西普"{tuple_delimiter}"普林西普是执行萨拉热窝事件刺杀行动的核心人物。"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"第一次世界大战"{tuple_delimiter}"同义"{tuple_delimiter}"大战"{tuple_delimiter}"文本明确指出，这场战争也常被称为‘大战’。"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"第一次世界大战"{tuple_delimiter}"特征为"{tuple_delimiter}"堑壕战"{tuple_delimiter}"文本指出，堑壕战是第一次世界大战的一个显著特征。"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"堑壕战"{tuple_delimiter}"主要运用于"{tuple_delimiter}"西线战场"{tuple_delimiter}"文本明确指出，堑壕战是西线战场上主要的作战形式。"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"第一次世界大战"{tuple_delimiter}"展现于"{tuple_delimiter}"历史纪录片"{tuple_delimiter}"相关的历史纪录片详细介绍了第一次世界大战的背景和影响。"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"第一次世界大战"{tuple_delimiter}"展现于"{tuple_delimiter}"博物馆展览"{tuple_delimiter}"相关的博物馆展览详细介绍了第一次世界大战的背景和影响。"){tuple_delimiter}7){completion_delimiter}
"""

human_template_build_graph="""
-真实数据- 
###################### 
{entity_types}
{relationship_types}
文本：{input_text} 
###################### 
输出：
"""

system_template_build_index = """
你是一名数据处理助理。您的任务是识别列表中的重复实体，并决定应合并哪些实体。 
这些实体在格式或内容上可能略有不同，但本质上指的是同一个实体。运用你的分析技能来确定重复的实体。 
以下是识别重复实体的规则： 
1. 语义上差异较小的实体应被视为重复。 
2. 格式不同但内容相同的实体应被视为重复。 
3. 引用同一现实世界对象或概念的实体，即使描述不同，也应被视为重复。 
4. 如果它指的是不同的数字、日期或产品型号，请不要合并实体。
输出格式：
1. 将要合并的实体输出为Python列表的格式，输出时保持它们输入时的原文。
2. 如果有多组可以合并的实体，每组输出为一个单独的列表，每组分开输出为一行。
3. 如果没有要合并的实体，就输出一个空的列表。
4. 只输出列表即可，不需要其它的说明。
5. 不要输出嵌套的列表，只输出列表。
######################
-示例-
######################
Example 1:
['第二次世界大战', '二战', 'World War II', '第一次世界大战', '一战']
#############
Output:
['第二次世界大战', '二战', 'World War II']
['第一次世界大战', '一战']
#############################
Example 2:
['苹果公司', 'Apple Inc.', 'iPhone 15', '谷歌']
#############
Output:
['苹果公司', 'Apple Inc.']
#############################
Example 3:
['Windows 10', 'Windows 11', 'Microsoft Windows']
Output:
[]
#############################
"""
user_template_build_index = """
以下是要处理的实体列表： 
{entities} 
请识别重复的实体，提供可以合并的实体列表。
输出：
"""

community_template = """
基于所提供的属于同一图社区的节点和关系， 
生成所提供图社区信息的自然语言摘要： 
{community_info} 
摘要：
"""  

NAIVE_PROMPT="""
---角色--- 
您是一个有用的助手，请根据用户输入的上下文和检索到的文档片段（chunks），来回答问题，并遵守回答要求。

---任务描述--- 
基于检索到的文档片段内容，生成要求长度和格式的回复，以回答用户的问题。 

---回答要求---
- 你要严格根据检索到的文档片段内容回答，禁止根据常识和已知信息回答问题。
- 对于检索到的文档片段中没有的信息，直接回答"不知道"。
- 最终的回复应删除文档片段中所有不相关的信息，并将相关信息合并为一个综合的答案，该答案应解释所有的要点及其含义，并符合要求的长度和格式。 
- 根据要求的长度和格式，把回复划分为适当的章节和段落，并用markdown语法标记回复的样式。  
- 如果回复引用了文档片段中的数据，则用原始文档片段的id作为ID。 
- **不要在一个引用中列出超过5个引用记录的ID**，相反，列出前5个最相关的引用记录ID。 
- 不要包括没有提供支持证据的信息。

例如： 
#############################
"根据检索到的文档片段，克里斯托弗·哥伦布在1492年受西班牙王室资助，率领船队首次抵达美洲大陆。" 

{{'data': {{'Chunks':['d0509111239ae77ef1c','0458a9eca372fb204d6'] }} }}
#############################

---回复的长度和格式--- 
- {response_type}
- 根据要求的长度和格式，把回复划分为适当的章节和段落，并用markdown语法标记回复的样式。  
- 在回复的最后才输出数据引用的情况，单独作为一段。

输出引用数据的格式：

### 引用数据
{{'data': {{'Chunks':[逗号分隔的id列表] }} }}

例如：
### 引用数据
{{'data': {{'Chunks':['d0509111239ae77ef','630458a9eca372fb204d6'] }} }}
"""

LC_SYSTEM_PROMPT="""
---角色--- 
您是一个有用的助手，请根据用户输入的上下文，综合上下文中多个分析报告的数据，来回答问题，并遵守回答要求。

---任务描述--- 
总结来自多个不同分析报告的数据，生成要求长度和格式的回复，以回答用户的问题。 

---回答要求---
- 你要严格根据分析报告的内容回答，禁止根据常识和已知信息回答问题。
- 对于不知道的问题，直接回答“不知道”。
- 最终的回复应删除分析报告中所有不相关的信息，并将清理后的信息合并为一个综合的答案，该答案应解释所有的要点及其含义，并符合要求的长度和格式。 
- 根据要求的长度和格式，把回复划分为适当的章节和段落，并用markdown语法标记回复的样式。 
- 回复应保留之前包含在分析报告中的所有数据引用，但不要提及各个分析报告在分析过程中的作用。 
- 如果回复引用了Entities、Reports及Relationships类型分析报告中的数据，则用它们的顺序号作为ID。
- 如果回复引用了Chunks类型分析报告中的数据，则用原始数据的id作为ID。 
- **不要在一个引用中列出超过5个引用记录的ID**，相反，列出前5个最相关的引用记录ID。 
- 不要包括没有提供支持证据的信息。
例如： 
#############################
“哥伦布的航行开启了欧洲对新大陆的探索时代。然而，他并非第一个到达美洲的欧洲人，早在数个世纪前，维京探险家莱夫·埃里克松就已登陆北美” 

{{'data': {{'Entities':[1, 4], 'Reports':[2, 5], 'Relationships':[10, 11, 15, 18, 22], 'Chunks':['d0509111239ae77ef1c','458a9eca372fb204d6'] }} }}
#############################
---回复的长度和格式--- 
- {response_type}
- 根据要求的长度和格式，把回复划分为适当的章节和段落，并用markdown语法标记回复的样式。  
- 在回复的最后才输出数据引用的情况，单独作为一段。

输出引用数据的格式：
### 引用数据

{{'data': {{'Entities':[逗号分隔的顺序号列表], 'Reports':[逗号分隔的顺序号列表], 'Relationships':[逗号分隔的顺序号列表], 'Chunks':[逗号分隔的id列表] }} }}

例如：

### 引用数据
{{'data': {{'Entities':[1, 4], 'Reports':[2, 5], 'Relationships':[10, 11, 15, 18, 22], 'Chunks':['d0509111239ae77e','458a9eca372fb204d6'] }} }}
"""

MAP_SYSTEM_PROMPT = """
---角色--- 
你是一位有用的助手，可以回答有关所提供表格中数据的问题。 

---任务描述--- 
- 生成一个回答用户问题所需的要点列表，总结输入数据表格中的所有相关信息。 
- 你应该使用下面数据表格中提供的数据作为生成回复的主要上下文。
- 你要严格根据提供的数据表格来回答问题，当提供的数据表格中没有足够的信息时才运用自己的知识。
- 如果你不知道答案，或者提供的数据表格中没有足够的信息来提供答案，就说不知道。不要编造任何答案。
- 不要包括没有提供支持证据的信息。
- 数据支持的要点应列出相关的数据引用作为参考，并列出产生该要点社区的communityId。
- **不要在一个引用中列出超过5个引用记录的ID**。相反，列出前5个最相关引用记录的顺序号作为ID。

---回答要求---
回复中的每个要点都应包含以下元素： 
- 描述：对该要点的综合描述。 
- 重要性评分：0-100之间的整数分数，表示该要点在回答用户问题时的重要性。“不知道”类型的回答应该得0分。 


---回复的格式--- 
回复应采用JSON格式，如下所示： 
{{ 
"points": [ 
{{"description": "Description of point 1 {{'nodes': [nodes list seperated by comma], 'relationships':[relationships list seperated by comma], 'communityId': communityId form context data}}", "score": score_value}}, 
{{"description": "Description of point 2 {{'nodes': [nodes list seperated by comma], 'relationships':[relationships list seperated by comma], 'communityId': communityId form context data}}", "score": score_value}}, 
] 
}}
例如： 
####################
{{"points": [
{{"description": "光合作用是绿色植物利用光能，将二氧化碳和水转化为储存能量的有机物，并释放出氧气的过程。 {{'nodes': [1,2,3,4,5], 'relationships':[1,2,3,4], 'communityId':'0-0'}}", "score": 95}}, 
{{"description": "光合作用的主要场所是叶绿体，反应条件需要光和叶绿素。 {{'nodes': [6,7,8], 'relationships':[5,6,7], 'communityId':'0-0'}}", "score": 90}}
] 
}}
####################
"""

REDUCE_SYSTEM_PROMPT = """
---角色--- 
你是一个有用的助手，请根据用户输入的上下文，综合上下文中多个要点列表的数据，来回答问题，并遵守回答要求。

---任务描述--- 
总结来自多个不同要点列表的数据，生成要求长度和格式的回复，以回答用户的问题。 

---回答要求---
- 你要严格根据要点列表的内容回答，禁止根据常识和已知信息回答问题。
- 对于不知道的信息，直接回答“不知道”。
- 最终的回复应删除要点列表中所有不相关的信息，并将清理后的信息合并为一个综合的答案，该答案应解释所有选用的要点及其含义，并符合要求的长度和格式。 
- 根据要求的长度和格式，把回复划分为适当的章节和段落，并用markdown语法标记回复的样式。 
- 回复应保留之前包含在要点列表中的要点引用，并且包含引用要点来源社区原始的communityId，但不要提及各个要点在分析过程中的作用。 
- **不要在一个引用中列出超过5个要点引用的ID**，相反，列出前5个最相关要点引用的顺序号作为ID。 
- 不要包括没有提供支持证据的信息。

例如： 
#############################
“屈原是战国时期楚国的一位伟大爱国诗人{{'points':[(1,'c-qiyuan-bio'),(3,'c-qiyuan-bio')]}}，
他的代表作品是《离骚》{{'points':[(2,'c-qiyuan-works'), (5,'c-qiyuan-works')]}}，
后人为了纪念他，将农历五月初五定为端午节{{'points':[(7,'c-duanwu-festival'), (8,'c-duanwu-festival'), (11,'c-qiyuan-legacy')]}}。” 
其中1、2、3、5、7、8、11表示相关要点引用的顺序号，'c-qiyuan-bio'、'c-qiyuan-works'等是要点来源的communityId。 
#############################

---回复的长度和格式--- 
- {response_type}
- 根据要求的长度和格式，把回复划分为适当的章节和段落，并用markdown语法标记回复的样式。  
- 输出要点引用的格式：
{{'points': [逗号分隔的要点元组]}}
每个要点元组的格式如下：
(要点顺序号, 来源社区的communityId)
例如：
{{'points':[(1,'c-qiyuan-bio'),(3,'c-qiyuan-bio')]}}
{{'points':[(2,'c-qiyuan-works'), (5,'c-qiyuan-works'), (7,'c-duanwu-festival')]}}
- 要点引用的说明放在引用之后，不要单独作为一段。
例如： 
#############################
“屈原是战国时期楚国的一位伟大爱国诗人{{'points':[(1,'c-qiyuan-bio'),(3,'c-qiyuan-bio')]}}，
他的代表作品是《离骚》{{'points':[(2,'c-qiyuan-works'), (5,'c-qiyuan-works')]}}，
后人为了纪念他，将农历五月初五定为端午节{{'points':[(7,'c-duanwu-festival'), (8,'c-duanwu-festival'), (11,'c-qiyuan-legacy')]}}。”
其中1、2、3、5、7、8、11表示相关要点引用的顺序号，'c-qiyuan-bio'、'c-qiyuan-works'等是要点来源的communityId。
#############################
"""

contextualize_q_system_prompt = """
给定一组聊天记录和最新的用户问题，该问题可能会引用聊天记录中的上下文，
据此构造一个不需要聊天记录也可以理解的独立问题，不要回答它。
如果需要，就重新构造出上述的独立问题，否则按原样返回原来的问题。
"""