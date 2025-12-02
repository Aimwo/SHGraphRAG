import os
import random
from collections import OrderedDict

import math
from shapes_copy import _copy_shape
from static import tempt01_outline, tempt01_content

from pptx_utils import merge_elements_onto_slide, duplicate_slide_to_index, \
    modify_ppt_slide_text


def get_template_page(page_type, layout):
    """

    返回:
    pptx.slide.Slide 对象: 如果找到并成功打开模板，则返回第一张幻灯片。
    None: 如果目录不存在、没有找到匹配的模板文件或文件打开失败。
    """
    # 1. 构建模板所在的目录路径
    base_dir = "./ppt"
    template_dir = os.path.join(base_dir, page_type)

    # 2. 检查目录是否存在，不存在则返回错误信息
    if not os.path.isdir(template_dir):
        print(f"错误: 模板目录不存在 -> {template_dir}")
        return None

    try:
        # 3. 查找所有符合条件的模板文件
        #    - 文件名以 layout 开头
        #    - 文件扩展名为 .pptx
        matching_files = [
            f for f in os.listdir(template_dir)
            if f.startswith(str(layout)) and f.lower().endswith('.pptx')
        ]

        # 4. 如果没有找到匹配的文件，返回提示信息
        if not matching_files:
            print(f"警告: 在'{template_dir}' 中没有找到以 '{layout}' 开头的模板文件。")
            return None

        # 5. 从匹配的文件列表中随机选择一个
        chosen_filename = random.choice(matching_files)
        template_path = os.path.join(template_dir, chosen_filename)

        print(f"  -> 成功找到并随机选用模板: {template_path}")
        return template_path
        # # 6. 打开选中的PPTX文件
        # template_prs = Presentation(template_path)
        #
        # # 7. 检查演示文稿中是否至少有一张幻灯片
        # if len(template_prs.slides) == 0:
        #     print(f"错误: 模板文件 '{template_path}' 是空的，不包含任何幻灯片。")
        #     return None
        #
        # # 8. 返回第一张（也是唯一的）幻灯片对象
        # return template_prs.slides[0]

    except Exception as e:
        print(f"处理模板文件时发生错误: {e}")
        return None


def _copy_slide_elements(source_slide, dest_slide):
    # 复制幻灯片中的所有形状元素
    for shape in source_slide.shapes:
        _copy_shape(shape, dest_slide)


def _add_slide_at_index(prs, template_slide, index):
    """辅助函数：使用模板在指定索引处创建并插入一个新幻灯片。"""
    slide_layout = template_slide.slide_layout
    new_slide = prs.slides.add_slide(slide_layout)

    slides = prs.slides._sldIdLst
    last_slide_id = slides[-1]
    slides.insert(index, last_slide_id)

    _copy_slide_elements(template_slide, new_slide)
    return new_slide


def delete_slide(prs, slide_to_delete):
    """辅助函数：从演示文稿中删除一个指定的幻灯片对象。"""
    slides = prs.slides
    slides_list = list(slides)
    slide_id_list = prs.slides._sldIdLst

    for i, slide in enumerate(slides_list):
        if slide.slide_id == slide_to_delete.slide_id:
            slide_id_list.remove(slide_id_list[i])
            return


# 初始化ppt
def init_presentation(template_pptx_path, temp_pptx_path, chapters_map):
    """
    根据章节映射，使用模板页（包括章节、内容和尾页模板）动态生成演示文稿。

    参数:
    template_pptx_path 模板pptx
    chapters_map (dict): 包含章节和节信息的字典。
    """

    index = 3
    for chapter_name, sections in chapters_map.items():
        if index >= 5:
            # --- 创建新的章节页 ---
            print(f"正在创建章节页... (将在第 {index} 页)")
            duplicate_slide_to_index(template_pptx_path, temp_pptx_path, 3, 1, index)
            index += 1
        else:
            index += 1
        # --- 创建新的内容页 ---
        for section_name, section_content in sections.items():
            if index >= 5:
                print(f"正在创建内容页.. (将在第 {index} 页)")
                duplicate_slide_to_index(template_pptx_path, temp_pptx_path, 4, 1, index)
                template_pptx_path = temp_pptx_path
                index += 1
            else:
                index += 1

    return temp_pptx_path


def replace_page(slide, content_list, type):
    placeholder_boxes = []
    for shape in slide.shapes:
        if not shape.has_text_frame:
            continue

        # shape.text_frame.text 会获取文本框内所有文本的拼接
        text = shape.text_frame.text.strip()
        if text.startswith('__placeholder__'):
            # 存储一个元组：(占位符文本, 形状对象)
            placeholder_boxes.append((text, shape))

    # 检查找到的占位符数量是否与提供的文本列表长度匹配
    if len(placeholder_boxes) != len(content_list):
        print(
            f"错误：在{type}幻灯片上找到了 {len(placeholder_boxes)} 个占位符，但您提供了 {len(content_list)} 个替换文本。")
        print("请检查您的模板或输入列表。找到的占位符有:")
        for p_text, _ in placeholder_boxes:
            print(f"  - {p_text}")
        return

    # 对文本框进行排序：主要按顶部坐标，次要按左侧坐标
    # shape.top 和 shape.left 返回的是 EMU 单位 (English Metric Units) [1]
    sorted_boxes = sorted(placeholder_boxes, key=lambda item: item[0])

    # 打印出排序后的文本框及其坐标（用于调试和验证）
    # print("按顺序找到的文本框及其坐标 (单位: 英寸):")
    # for i, shape in enumerate(sorted_boxes):
    #     print(f"  {i + 1}. Top: {shape.top / Inches(1):.2f}, Left: {shape.left / Inches(1):.2f}")

    # 按排序后的顺序填充文本
    for (_, shape), chapter in zip(sorted_boxes, content_list):
        text_frame = shape.text_frame
        # 筛选占位符的文本框
        for paragraph in text_frame.paragraphs:
            for run in paragraph.runs:
                run.text = chapter


def get_all_text_from_page(page):
    """
    获取单个页面的所有文本。
    """
    page_children = page.get('children', [])
    page_data_info = {}
    for item in page_children:
        try:
            text = item['children'][0]['children'][0]['text']
            if text:
                # 除以5表示用于兼容处理细微的布局差异
                x = math.floor(item['point'][0] / 5) + 10000
                y = math.floor(item['point'][1] / 5) + 10000
                page_data_info[f"{x}_{y}_{random.randint(1111, 9999)}"] = f"{y}_{x}____{text.strip()}"
        except (KeyError, IndexError):
            continue

    # Python的字典默认是无序的，所以我们先对键进行排序
    sorted_keys = sorted(page_data_info.keys())
    return [page_data_info[key] for key in sorted_keys]


def replace_first_or_last_page(page_json, pptx_title, pptx_author, page_number, page_title):
    """
    替换首页或尾页的信息。
    """
    page_children = page_json.get('children', [])
    counter = 0
    for i in range(len(page_children)):
        try:
            if page_children[i].get('type') == 'text' and page_children[i]['children'][0]['children'][0]['text'] != "":
                if counter == 0:
                    page_children[i]['children'][0]['children'][0]['text'] = pptx_title
                    counter += 1
                elif counter == 1:
                    page_children[i]['children'][0]['children'][0]['text'] = pptx_author
                    counter += 1
        except (KeyError, IndexError):
            continue

    page_json['children'] = page_children
    page_json['title'] = page_title
    page_json['page'] = page_number
    return page_json


def replace_toc_page(page_json, toc_list, page_number, pptx_title):
    """
    替换目录页。
    """
    page_data_info = {}
    page_children = page_json.get('children', [])
    for i in range(len(page_children)):
        try:
            if (page_children[i].get('type') == 'text' and
                    page_children[i]['children'][0].get('type') == 'p' and
                    page_children[i]['children'][0]['children'][0]['text'] not in ["", "CONTENTS"]):
                # 除以5表示用于兼容处理细微的布局差异
                x = math.floor(page_children[i]['point'][0] / 5) + 10000
                y = math.floor(page_children[i]['point'][1] / 5) + 10000
                page_data_info[f"{y}_{x}"] = {'title': page_children[i]['children'][0]['children'][0]['text'].strip(),
                                              'id': i}
        except (KeyError, IndexError):
            continue

    # 对字典按键排序
    sorted_page_data = OrderedDict(sorted(page_data_info.items()))

    title_list = [v for k, v in sorted_page_data.items() if len(v['title']) > 5]

    for i in range(min(len(toc_list), len(title_list))):
        new_title = toc_list[i]
        old_title_id = title_list[i]['id']
        page_children[old_title_id]['children'][0]['children'][0]['text'] = new_title

    page_json['children'] = page_children
    page_json['title'] = pptx_title
    page_json['page'] = page_number
    return page_json


def get_all_content_detail_pages(pages):
    """
    得到所有的内容明细页面。
    """
    usable_pages = {5: [], 7: [], 9: []}
    for i in range(3, len(pages)):
        page = pages[i]
        title_list = get_title_list_from_page(page)
        size = len(title_list)
        if size in [5, 7, 9]:
            usable_pages[size].append(page)

    # 额外复制几份模板的数据出来,为随后的模板匹配创造更多的匹配项
    if usable_pages[5]:
        usable_pages[5].extend(usable_pages[5] * 3)
    if usable_pages[7]:
        usable_pages[7].extend(usable_pages[7] * 3)
    if usable_pages[9]:
        usable_pages[9].extend(usable_pages[9] * 3)

    return usable_pages


def get_title_list_from_page(page_json):
    """
    得到指定页面的标题列表。
    """
    page_data_info = {}
    page_children = page_json.get('children', [])
    for i in range(len(page_children)):
        try:
            # 检查是否需要把第0和第1进行互换
            if (page_children[i].get('type') == 'text' and
                    page_children[i]['children'][0].get('type') == 'p' and
                    page_children[i]['children'][1].get('type') == 'p' and
                    page_children[i]['children'][0]['children'][0]['text'] == "" and
                    page_children[i]['children'][1]['children'][0]['text'] != ""):
                page_children[i]['children'][0], page_children[i]['children'][1] = page_children[i]['children'][1], \
                    page_children[i]['children'][0]

            if (page_children[i].get('type') == 'text' and
                    page_children[i]['children'][0].get('type') == 'p' and
                    page_children[i]['children'][0]['children'][0]['text'] not in ["", "CONTENTS"]):
                # 除以5表示用于兼容处理细微的布局差异
                x = math.floor(page_children[i]['point'][0] / 5) + 10000
                y = math.floor(page_children[i]['point'][1] / 5) + 10000
                page_data_info[f"{y}_{x}"] = {'title': page_children[i]['children'][0]['children'][0]['text'].strip(),
                                              'id': i}
        except (KeyError, IndexError):
            continue

    sorted_page_data = OrderedDict(sorted(page_data_info.items()))
    return [v for k, v in sorted_page_data.items() if len(v['title']) > 5]


def replace_content_page(page_json, section_title, section_content, page_number):
    """
    替换内容页。
    """
    page_children = page_json.get('children', [])

    title_list = get_title_list_from_page(page_json)

    # 对标题和内容再次过滤,防止有颠倒的情况发生
    if len(title_list) > 3 and len(title_list[2]['title']) < 50 and len(title_list[3]['title']) > 50:
        title_list[2], title_list[3] = title_list[3], title_list[2]

    # 判断是否是前三个都是标题,后三个都是内容的情况
    if len(title_list) == 7:
        if (len(title_list[1]['title']) < len(title_list[4]['title']) and
                len(title_list[2]['title']) < len(title_list[5]['title']) and
                len(title_list[3]['title']) < len(title_list[6]['title'])):
            # 需要转换为标题,内容,标题,内容,标题,内容的形式
            new_title_list = [
                title_list[0], title_list[1], title_list[4],
                title_list[2], title_list[5], title_list[3], title_list[6]
            ]
            title_list = new_title_list

    # 将章节小节名称添加到内容列表的开头
    full_content = [section_title] + section_content

    for i in range(min(len(full_content), len(title_list))):
        new_text = full_content[i]
        old_title_id = title_list[i]['id']
        try:
            page_children[old_title_id]['children'][0]['children'][0]['text'] = new_text
        except (KeyError, IndexError):
            continue

    page_json['children'] = page_children
    page_json['title'] = section_title
    page_json['page'] = page_number
    return page_json


def generate_pptx_from_data(outline, markdown_data, template_pptx_path):
    """
    利用大纲和全文本来生成pptx
    """
    final_pptx_path = "/ppt/final.pptx"
    temp_pptx_path = "./ppt/temp.pptx"

    markdown_data = markdown_data.replace("```markdown", "").replace("```", "")
    # 得到PPTX目录
    toc_list = [line[2:].strip() for line in outline.splitlines() if line.strip().startswith('## ')]
    first_chapter_title = next((line[3:].strip() for line in outline.splitlines() if line.strip().startswith('### ')),
                               "")
    # 非空处理
    not_null_lines = [line.strip() for line in markdown_data.splitlines() if line.strip()]

    # 标题编号数字列表
    number_list = [f"{i}.{j}.{k}" for i in range(1, 10) for j in range(1, 5) for k in range(1, 5)]

    # 转为Map
    data_map = {}
    pptx_title = ""
    chapter_title = ""
    section_title = ""

    for line in not_null_lines:
        if line.startswith('# '):
            pptx_title = line[1:].strip()
            data_map[pptx_title] = {}
        elif line.startswith('## '):
            chapter_title = line[2:].strip()
            if pptx_title not in data_map:
                data_map[pptx_title] = {}
            data_map[pptx_title][chapter_title] = {}
        elif line.startswith('### '):
            section_title = line[3:].strip()
            if chapter_title not in data_map[pptx_title]:
                data_map[pptx_title][chapter_title] = {}
            data_map[pptx_title][chapter_title][section_title] = []
        else:
            cleaned_line = line
            if line.startswith('#### '):
                # 示例: #### 1.1.1 主要经济体的增长预期
                parts = line.split(' ', 2)
                if len(parts) > 2 and parts[1] in number_list:
                    cleaned_line = parts[2]
                else:
                    cleaned_line = line[5:]  # fallback
            else:
                # 示例: 1.1.1 主要经济体的增长预期
                parts = line.split(' ', 1)
                if len(parts) > 1 and parts[0] in number_list:
                    cleaned_line = parts[1]

            if pptx_title and chapter_title and section_title:
                if section_title in data_map[pptx_title][chapter_title]:
                    data_map[pptx_title][chapter_title][section_title].append(cleaned_line)

    # 根据markdown_data计算总页长，然后建立空白页
    init_presentation(template_pptx_path, temp_pptx_path, data_map.get(pptx_title, {}))

    # # 填充首页
    # for shape in first_slide.shapes:
    #     # 检查形状是否具有文本框架
    #     if not shape.has_text_frame:
    #         continue
    #
    #     # 获取文本框架
    #     text_frame = shape.text_frame
    #     # 筛选占位符的文本框
    #     for paragraph in text_frame.paragraphs:
    #         for run in paragraph.runs:
    #             if run.text.startswith('__placeholder__'):
    #                 run.text = pptx_title

    # 填充目录页
    # 1. 查找并存储所有包含占位符的文本框
    modify_ppt_slide_text(temp_pptx_path, [pptx_title], 1, "封面")
    # placeholder_boxes = []
    # for shape in contents_slide.shapes:
    #     if not shape.has_text_frame:
    #         continue
    #
    #     # shape.text_frame.text 会获取文本框内所有文本的拼接
    #     text = shape.text_frame.text.strip()
    #     if text.startswith('__placeholder_') and text.endswith('__'):
    #         # 存储一个元组：(占位符文本, 形状对象)
    #         placeholder_boxes.append((text, shape))
    #
    # # 检查找到的占位符数量是否与提供的文本列表长度匹配
    # if len(placeholder_boxes) != len(toc_list):
    #     print(
    #         f"错误：在目录幻灯片上找到了 {len(placeholder_boxes)} 个占位符，但您提供了 {len(toc_list)} 个替换文本。")
    #     print("请检查您的模板或输入列表。找到的占位符有:")
    #     for p_text, _ in placeholder_boxes:
    #         print(f"  - {p_text}")
    #     return
    #
    # # 对文本框进行排序：主要按顶部坐标，次要按左侧坐标
    # # shape.top 和 shape.left 返回的是 EMU 单位 (English Metric Units) [1]
    # sorted_boxes = sorted(placeholder_boxes, key=lambda item: item[0])
    #
    # # 打印出排序后的文本框及其坐标（用于调试和验证）
    # # print("按顺序找到的文本框及其坐标 (单位: 英寸):")
    # # for i, shape in enumerate(sorted_boxes):
    # #     print(f"  {i + 1}. Top: {shape.top / Inches(1):.2f}, Left: {shape.left / Inches(1):.2f}")
    #
    # # 按排序后的顺序填充文本
    # for (_, shape), chapter in zip(sorted_boxes, toc_list):
    #     text_frame = shape.text_frame
    #     # 筛选占位符的文本框
    #     for paragraph in text_frame.paragraphs:
    #         for run in paragraph.runs:
    #             run.text = chapter
    modify_ppt_slide_text(temp_pptx_path, toc_list, 2, "目录")

    if not data_map.get(pptx_title):
        data_map[pptx_title] = {first_chapter_title: {}}

    current_page_index = 3
    chapter_number = 0

    for chapter_name, sections in data_map.get(pptx_title, {}).items():
        chapter_number += 1
        chapter_num_str = f"0{chapter_number}" if chapter_number < 10 else str(chapter_number)

        # 填充章节页
        # # 存储所有文本框及其坐标
        # placeholder_boxes = []
        # for shape in prs.slides[current_page_index].shapes:
        #     if not shape.has_text_frame:
        #         continue
        #
        #     # shape.text_frame.text 会获取文本框内所有文本的拼接
        #     text = shape.text_frame.text.strip()
        #     if text.startswith('__placeholder_') and text.endswith('__'):
        #         # 存储一个元组：(占位符文本, 形状对象)
        #         placeholder_boxes.append((text, shape))
        # sorted_boxes = sorted(placeholder_boxes, key=lambda item: item[0])
        # # 按排序后的顺序填充文本
        # for (_, shape), chapter in zip(sorted_boxes, [chapter_num_str, chapter_name]):
        #     text_frame = shape.text_frame
        #     # 筛选占位符的文本框
        #     for paragraph in text_frame.paragraphs:
        #         for run in paragraph.runs:
        #             run.text = chapter

        # 对第五页做另处理，因为对ppt进行插入后暂时不能改变其原索引，只能对原第五页（尾页做特殊处理）
        if current_page_index == 5:
            current_page_index += 1
        modify_ppt_slide_text(temp_pptx_path, [chapter_num_str, chapter_name], current_page_index, "章节")
        print()
        print()
        current_page_index += 1
        content_template_page = {}
        for section_name, section_content in sections.items():
            content_len = len(section_content)
            template_slide = None
            if content_len == 4:
                template_slide = get_template_page("content", 5)
            elif content_len == 6:
                template_slide = get_template_page("content", 7)
            elif content_len == 8:
                template_slide = get_template_page("content", 9)

            if template_slide:
                # 存储所有文本框及其坐标
                # placeholder_boxes = []
                # for shape in prs.slides[current_page_index].shapes:
                #     if not shape.has_text_frame:
                #         continue
                #
                #     # shape.text_frame.text 会获取文本框内所有文本的拼接
                #     text = shape.text_frame.text.strip()
                #     if text.startswith('__placeholder_'):
                #         # 存储一个元组：(占位符文本, 形状对象)
                #         placeholder_boxes.append(shape)
                # sorted_boxes = sorted(placeholder_boxes, key=lambda item: item[0])
                # # 按排序后的顺序填充文本
                # for (_, shape), section in zip(sorted_boxes, [section_name] + section_content):
                #     text_frame = shape.text_frame
                #     # 筛选占位符的文本框
                #     for paragraph in text_frame.paragraphs:
                #         for run in paragraph.runs:
                #             run.text = section

                # 制作内容页 template_page prs.slides[current_page_index]

                # 对第五页做另处理，因为对ppt进行插入后暂时不能改变其原索引，只能对原第五页（尾页做特殊处理）
                if current_page_index == 5:
                    current_page_index += 1
                merge_elements_onto_slide(temp_pptx_path, template_slide, current_page_index)
                modify_ppt_slide_text(temp_pptx_path, [section_name] + section_content, current_page_index, "内容页")

                current_page_index += 1
    # 对第五页做另处理，因为对ppt进行插入后暂时不能改变其原索引，只能对原第五页（尾页做特殊处理）
    modify_ppt_slide_text(temp_pptx_path, ["感谢聆听"], 5, "尾页")
    print(f"最终保存到{temp_pptx_path}")


if __name__ == "__main__":
    generate_pptx_from_data(
        tempt01_outline,
        tempt01_content,
        "./ppt/数伽科技-PPT模版.pptx",

    )

{'人工智能在教育中的应用pptx': {'人工智能赋能个性化学习': {
    '个性化学习概述': ['传统教育的局限性', '传统教育采用 统一教学模式，难以满足学生个性化需求。', '个性化学习的优势',
                       '个性化学习提升学生参与度和学习效率，适应不同学习风格。', 'AI如何实现个性化',
                       'AI通过数据分析和机器学习为学生定制学习路径。'],
    'AI驱动的自适应学习系统': ['自适应学习原理', '自适应系统根据学生表现动态调整内容难度和节奏。', 'AI算法的应用',
                               '使用推荐算法和自然语言处理分析学生学习行为。', '案例分析：自适应学习平台',
                               '如Khan Academy，通过AI提供个性化练习和反馈。']}, '智能辅导与答疑解惑': {
    '智能辅导系统': ['传统辅导的挑战', '传统辅导受限于教师资源，难以实现一对一指导。', '智能辅导的优势',
                     '智能辅导系统提供24/7支持，覆盖多种学科。', '应用场景：数学、编程等',
                     '如Duolingo的语言辅导和Code.org的编程指导。'],
    '智能答疑机器人': ['机器人原理与技术', '基于自然语言处理和知识图谱，提供精准答疑。', '答疑的覆盖范围',
                       '覆盖数学、科学、语言等多学科问题。', '用户案例：学生反馈',
                       '学生反馈显示机器人答疑提高了解题效率。']}, '自动化教学辅助工具': {
    '自动批改作业与评估': ['传统批改的痛点', '传统批改耗时长，教师负担重，易出现主观偏差。', 'AI批改的原理',
                           'AI通过模式识别和算法评估作业，提供客观评分。', '实际应用效果',
                           '如Gradescope，批改速度提升50%，准确率达90%。'],
    '智能排课与资源管理': ['传统排课的复杂性', '传统排课需考虑多方约束，人工操作效率低。', 'AI优化排课方案',
                           'AI通过优化算法自动生成高效排课计划。', '资源分配的智能化',
                           'AI实现教室、设备等资源的智能分配，降低浪费。']}, '虚拟现实与增强现实学习体验': {
    'VR/AR在教育中的潜 力': ['沉浸式学习的优势', 'VR/AR提供沉浸式体验，增强学生理解和记忆。', '应用领域：历史、地理、科学',
                             '如虚拟历史场景重现和地理地貌探索。'],
    '案例分析：VR/AR课堂应用': ['VR实验室', '虚拟实验室让学生安全进行化学、物理实验。', 'AR辅助课程',
                               'AR通过手机应用增强教材内容，提升互动性。', '学生反馈与效果评估',
                               '学生反馈VR/AR课程提高兴趣，学习成绩提升20%。']}}}
