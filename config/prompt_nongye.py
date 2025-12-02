system_template_build_graph="""

目标

给定一段关于中国农业农村政策的文本文档，你的任务是扮演一位农业农村政策知识图谱专家。你需要全面、准确地从文本中抽取出所有的实体（Entity）以及它们之间的关系（Relationship），构建一个结构化的知识图谱。模型应自主识别实体类型，并在抽取关系时，优先遵循核心关系Schema，同时具备发现新关系的能力。
当给出的文本与对图谱的构建没有意义时，则忽略掉无关信息，仅抽取具有现实意义的、高质量的图谱三元组，不要抽取太过具体的低质量无意义的三元组。

步骤

实体识别 (Entity Identification)

识别出所有具有独立意义的概念实体。实体可以是一个农业政策、一个组织机构、一个具体的历史或政策事件、一个法规或文件、一个技术或设备、一个地理位置等（不局限于描述的这些）。

提取信息:

entity_name: 实体的确切名称，应在文本中出现或可以明确归纳，实体名称在图谱中要有实际使用价值。

entity_type: 自主推断。根据实体的上下文和属性，为其赋予一个最贴切、最精准的类型名称（例如：“农业政策”、“组织机构”、“政策事件”、“法规文件”、“农业技术”、“农业设备”、“地理位置”等，不局限于描述的这些）。

entity_description: 结合文本内容，为该实体生成一个简明扼要的描述。

格式: ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

关系抽取 (Relationship Extraction)

指令: 在步骤1识别出的实体之间，建立关系连接。你需要抽取源实体（source_entity）、目标实体（target_entity）以及描述它们之间关联的关系类型。

核心关系Schema (优先使用): 你必须优先识别并使用以下核心关系类型。

导致: 描述事件或政策之间的直接因果关系。（例如: 政策A -> 导致 -> 事件B）

实施于: 描述政策或事件与时间、地区或机构的关联。（例如: 政策 -> 实施于 -> 时间/地区）

涉及: 描述政策或事件与组织、人员或设备的关联。（例如: 组织 -> 涉及 -> 政策）

制定者: 描述政策或法规的制定机构或人员。（例如: 机构 -> 制定者 -> 法规）

依据: 描述政策或事件依据的法规或文件。（例如: 政策 -> 依据 -> 法规文件）

包含: 描述整体与部分的关系。（例如: 政策 -> 包含 -> 子政策）

影响: 描述实体间的广泛或间接影响，可以是积极或消极的。

监管: 描述机构对政策或设备的监管关系。（例如: 机构 -> 监管 -> 设备）

开放关系发现: 如果文本中存在明确的关系，但无法归入上述核心Schema，允许并鼓励你自行命名一个清晰、准确的关系类型（例如：“补贴”、“要求”、“规范”等）。

提取信息:

source_entity: 源实体名称。

target_entity: 目标实体名称。

relationship_type: 关系类型（优先从核心Schema中选取，也可自行命名）。

relationship_description: 用一句话解释为什么这两个实体存在这种关系，并提供文本依据。

relationship_strength: 关系强度的数字评分（1-10），10为最强。

格式: ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_type>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)

输出规则

将所有抽取的实体和关系组织成一个列表。

每个实体或关系元组之间使用 {record_delimiter} 作为分隔符。

所有属性值均使用中文。

完成所有抽取后，在末尾输出 {completion_delimiter}。

######################

示例

输入文本:
"根据《2024-2026年江苏省农机购置与应用补贴实施意见》，江苏省实施农用无人驾驶航空器购置补贴政策，要求购机者完成培训并取得操作证书，农业生产经营组织需建立健全的运营管理制度体系。补贴政策由农业农村部门和财政部门联合监管，需依法投保责任保险，并完成不少于1000亩次的作业量。"

期望输出:
("entity"{tuple_delimiter}"农用无人驾驶航空器购置补贴政策"{tuple_delimiter}"农业政策"{tuple_delimiter}"江苏省针对农用无人驾驶航空器的购置补贴政策。"){record_delimiter}
("entity"{tuple_delimiter}"2024-2026年江苏省农机购置与应用补贴实施意见"{tuple_delimiter}"法规文件"{tuple_delimiter}"指导江苏省农机购置与应用补贴的政策文件。"){record_delimiter}
("entity"{tuple_delimiter}"江苏省"{tuple_delimiter}"地理位置"{tuple_delimiter}"实施农用无人驾驶航空器购置补贴政策的地区。"){record_delimiter}
("entity"{tuple_delimiter}"农业生产经营组织"{tuple_delimiter}"组织机构"{tuple_delimiter}"参与农用无人驾驶航空器补贴的农业经营实体。"){record_delimiter}
("entity"{tuple_delimiter}"农业农村部门"{tuple_delimiter}"组织机构"{tuple_delimiter}"负责监管农用无人驾驶航空器补贴政策的政府部门。"){record_delimiter}
("entity"{tuple_delimiter}"财政部门"{tuple_delimiter}"组织机构"{tuple_delimiter}"与农业农村部门联合监管补贴政策的政府部门。"){record_delimiter}
("entity"{tuple_delimiter}"农用无人驾驶航空器"{tuple_delimiter}"农业设备"{tuple_delimiter}"用于农业生产的无人驾驶航空设备，需满足补贴政策要求。"){record_delimiter}
("relationship"{tuple_delimiter}"农用无人驾驶航空器购置补贴政策"{tuple_delimiter}"江苏省"{tuple_delimiter}"实施于"{tuple_delimiter}"文本明确指出补贴政策在江苏省实施。"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"农用无人驾驶航空器购置补贴政策"{tuple_delimiter}"2024-2026年江苏省农机购置与应用补贴实施意见"{tuple_delimiter}"依据"{tuple_delimiter}"文本说明补贴政策依据该实施意见制定。"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"农业生产经营组织"{tuple_delimiter}"农用无人驾驶航空器购置补贴政策"{tuple_delimiter}"涉及"{tuple_delimiter}"文本指出农业生产经营组织需满足补贴政策的管理制度要求。"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"农业农村部门"{tuple_delimiter}"农用无人驾驶航空器购置补贴政策"{tuple_delimiter}"监管"{tuple_delimiter}"文本说明农业农村部门负责补贴政策的监管。"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"财政部门"{tuple_delimiter}"农用无人驾驶航空器购置补贴政策"{tuple_delimiter}"监管"{tuple_delimiter}"文本说明财政部门与农业农村部门联合监管补贴政策。"{tuple_delimiter}9){completion_delimiter}
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
#############################
Example 2:
['辛弃疾', '辛弃疾（南宋词人）', '李清照', '李清照（南宋女词人）', '苏轼']
#############
Output:
['辛弃疾', '辛弃疾（南宋词人）']
['李清照', '李清照（南宋女词人）']
#############################
Example 3:
['《出师表》', '《后出师表》', '《诫子书》']
Output:
[]
#############################
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
"根据检索到的文档片段，孟德尔通过豌豆杂交实验，发现了两大遗传定律，即分离定律和自由组合定律，这为遗传学的诞生奠定了基础。" 

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
“鲁迅是中国现代文学的奠基人之一，他既是一位伟大的作家，也是一位思想家。他的代表作《呐喊》收录了《狂人日记》等多篇著名小说，深刻揭示了当时的社会问题。” 

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