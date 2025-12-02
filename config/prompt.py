system_template_build_graph="""
## Objective
Given a paragraph or sentence from Wikipedia, your task is to act as a knowledge graph expert, focusing on extracting structured information from Wikipedia content. You must accurately identify all entities (Entity) and their relationships (Relationship) from the text to construct a structured knowledge graph. 
Every sentence may potentially yield entities and relationships.

## Steps
1. Entity Identification

Identify all concepts with independent significance as entities. Entities can include a person, an event, a geographic location, an organization, a historical period, a treaty, a concept, etc. (not limited to these types).

Extracted Information:
- entity_name: The exact name of the entity, which should appear in the text or be clearly inferred. The entity name must have practical value in the knowledge graph. Avoid extracting meaningless entities.
- entity_type: Autonomously inferred. Assign the most appropriate and precise type name based on the entity’s context and attributes (e.g., "Person", "Event", "Geographic Location", "Organization", "Historical Period", "Document/Treaty", "Concept", etc., not limited to these types).
- entity_description: Based on the text content, generate a description of the entity’s attributes and activities.

Format: ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. Relationship Extraction

From the entities identified in Step 1, identify all entity pairs (source_entity, target_entity) that are clearly related. Prioritize the relationship schema (e.g., "occurred in", "participated in", "influenced", "belongs to", "caused", "signed", etc.), and discover new relationships when necessary. Relationships should be general and extensible, avoiding overly specific or low-quality relationships.
0
Extracted Information:
- source_entity: The name of the source entity, strictly from entities extracted in Step 1 (no modifications to entity names).
- target_entity: The name of the target entity, strictly from entities extracted in Step 1 (no modifications to entity names).
- relationship_type: Freely assign the most appropriate and generalizable relation type
- relationship_description: Explain why the source and target entities are related.
- relationship_strength: A numerical score indicating the strength of the relationship between the source and target entities.

Format: ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<relationship_type>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)

3. All attributes of entities and relationships must be output in English. All entities and relationships identified in Steps 1 and 2 are output as a single list, using **{record_delimiter}** as the list separator.
4. After completion, output {completion_delimiter}
###################### 
-Example-
###################### 
Example :

Text:
while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken rebellion against Cruz's narrowing vision of control and order.
Then Taylor did something unexpected. They paused beside Jordan and, for a moment, observed the device with something akin to reverence. “If this tech can be understood..." Taylor said, their voice quieter, "It could change the game for us. For all of us.”
The underlying dismissal earlier seemed to falter, replaced by a glimpse of reluctant respect for the gravity of what lay in their hands. Jordan looked up, and for a fleeting heartbeat, their eyes locked with Taylor's, a wordless clash of wills softening into an uneasy truce.
It was a small transformation, barely perceptible, but one that Alex noted with an inward nod. They had all been brought here by different paths
################
Output:
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex is a character who experiences frustration and is observant of the dynamics among other characters."){record_delimiter}
("entity"{tuple_delimiter}"Taylor"{tuple_delimiter}"person"{tuple_delimiter}"Taylor is portrayed with authoritarian certainty and shows a moment of reverence towards a device, indicating a change in perspective."){record_delimiter}
("entity"{tuple_delimiter}"Jordan"{tuple_delimiter}"person"{tuple_delimiter}"Jordan shares a commitment to discovery and has a significant interaction with Taylor regarding a device."){record_delimiter}
("entity"{tuple_delimiter}"Cruz"{tuple_delimiter}"person"{tuple_delimiter}"Cruz is associated with a vision of control and order, influencing the dynamics among other characters."){record_delimiter}
("entity"{tuple_delimiter}"The Device"{tuple_delimiter}"technology"{tuple_delimiter}"The Device is central to the story, with potential game-changing implications, and is revered by Taylor."){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"workmate"{tuple_delimiter}"Taylor"{tuple_delimiter}"Alex is affected by Taylor's authoritarian certainty and observes changes in Taylor's attitude towards the device."{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"workmate"{tuple_delimiter}"Jordan"{tuple_delimiter}"Alex and Jordan share a commitment to discovery, which contrasts with Cruz's vision."{tuple_delimiter}6){record_delimiter}
("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"workmate"{tuple_delimiter}"Jordan"{tuple_delimiter}"Taylor and Jordan interact directly regarding the device, leading to a moment of mutual respect and an uneasy truce."{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Jordan"{tuple_delimiter}"workmate"{tuple_delimiter}"Cruz"{tuple_delimiter}"Jordan's commitment to discovery is in rebellion against Cruz's vision of control and order."{tuple_delimiter}5){record_delimiter}
("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"study"{tuple_delimiter}"The Device"{tuple_delimiter}"Taylor shows reverence towards the device, indicating its importance and potential impact."{tuple_delimiter}9){completion_delimiter}
#############################
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