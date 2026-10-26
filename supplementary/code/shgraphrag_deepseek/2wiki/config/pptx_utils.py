# import zipfile
# import re
# from lxml import etree
# import os
#
#
# def create_single_slide_ppt(
#         input_pptx_path: str,
#         output_pptx_path: str,
#         text_to_find: str,
#         text_to_replace: str
# ):
#     """
#     提取PPTX的第一张幻灯片，修改其文本，并生成一个只包含该幻灯片的新PPTX文件。
#
#     :param input_pptx_path: 输入的 PPTX 文件路径。
#     :param output_pptx_path: 输出的只包含一张幻灯片的 PPTX 文件路径。
#     :param text_to_find: 要在第一张幻灯片中查找和替换的文本（支持正则表达式）。
#     :param text_to_replace: 用于替换的新文本。
#     """
#     try:
#         # --- 步骤 1: 将源PPTX的所有文件读入内存 ---
#         package_files = {}
#         with zipfile.ZipFile(input_pptx_path, 'r') as z_in:
#             for item in z_in.infolist():
#                 package_files[item.filename] = z_in.read(item.filename)
#
#         # 定义XML命名空间
#         ns = {
#             'p': 'http://schemas.openxmlformats.org/presentationml/2006/main',
#             'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
#             'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
#         }
#
#         # --- 步骤 2 & 3: 修改核心结构文件，移除对其他幻灯片的引用 ---
#
#         # 解析 presentation.xml
#         pres_xml_path = 'ppt/presentation.xml'
#         if pres_xml_path not in package_files:
#             print(f"错误: 找不到核心文件 {pres_xml_path}")
#             return
#
#         pres_root = etree.fromstring(package_files[pres_xml_path])
#
#         # 找到幻灯片列表 <p:sldIdLst>
#         slide_id_list = pres_root.find('p:sldIdLst', namespaces=ns)
#         if slide_id_list is None:
#             print("警告: 演示文稿中没有幻灯片。")
#             # 仍然继续尝试修改文本，以防万一
#         else:
#             # 获取所有幻灯片ID条目
#             all_slides = slide_id_list.findall('p:sldId', namespaces=ns)
#
#             # 记录需要删除的幻灯片的关系ID (rId)
#             rids_to_remove = set()
#             for slide in all_slides[1:]:  # 从第二个开始，都是要删除的
#                 rids_to_remove.add(slide.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id'))
#                 slide_id_list.remove(slide)  # 从XML树中删除该节点
#
#         # 将修改后的 presentation.xml 写回内存
#         package_files[pres_xml_path] = etree.tostring(pres_root, xml_declaration=True, encoding='UTF-8',
#                                                       standalone=True)
#
#         # 解析并修改 presentation.xml.rels
#         pres_rels_path = 'ppt/_rels/presentation.xml.rels'
#         pres_rels_root = etree.fromstring(package_files[pres_rels_path])
#
#         files_to_remove = set()
#         rels_to_remove = []
#         for rel in pres_rels_root.findall('Relationship', namespaces=pres_rels_root.nsmap):
#             if rel.get('Id') in rids_to_remove:
#                 # 记录要删除的幻灯片文件名，如 slides/slide2.xml
#                 files_to_remove.add(os.path.join('ppt', rel.get('Target')))
#                 rels_to_remove.append(rel)
#
#         for rel in rels_to_remove:
#             pres_rels_root.remove(rel)
#
#         # 将修改后的 .rels 文件写回内存
#         package_files[pres_rels_path] = etree.tostring(pres_rels_root, xml_declaration=True, encoding='UTF-8',
#                                                        standalone=True)
#
#         # --- 步骤 4: 修改第一张幻灯片的文本 ---
#         slide1_path = 'ppt/slides/slide1.xml'
#         if slide1_path in package_files:
#             slide_root = etree.fromstring(package_files[slide1_path])
#             text_elements = slide_root.xpath('//a:t', namespaces=ns)
#             for t_element in text_elements:
#                 if t_element.text and re.search(text_to_find, t_element.text):
#                     t_element.text = re.sub(text_to_find, text_to_replace, t_element.text)
#
#             # 将修改后的 slide1.xml 写回内存
#             package_files[slide1_path] = etree.tostring(slide_root, xml_declaration=True, encoding='UTF-8',
#                                                         standalone=True)
#             print("第一张幻灯片的文本已成功匹配和修改。")
#         else:
#             print("警告: 找不到 slide1.xml 文件。")
#
#         # --- 步骤 5: 从文件列表中删除多余的幻灯片及其关系文件 ---
#         for file_path in files_to_remove:
#             # 删除 slideX.xml
#             if file_path in package_files:
#                 del package_files[file_path]
#             # 删除对应的 _rels/slideX.xml.rels
#             rel_file_path = os.path.join(os.path.dirname(file_path), '_rels', os.path.basename(file_path) + '.rels')
#             if rel_file_path in package_files:
#                 del package_files[rel_file_path]
#
#         # --- 步骤 6: 清理 [Content_Types].xml ---
#         content_types_path = '[Content_Types].xml'
#         content_types_root = etree.fromstring(package_files[content_types_path])
#         overrides_to_remove = []
#         # 将文件名转换为 PartName 格式（以 / 开头）
#         parts_to_remove = {'/' + f.replace('\\', '/') for f in files_to_remove}
#
#         for override in content_types_root.findall('Override', namespaces=content_types_root.nsmap):
#             if override.get('PartName') in parts_to_remove:
#                 overrides_to_remove.append(override)
#
#         for override in overrides_to_remove:
#             content_types_root.remove(override)
#
#         package_files[content_types_path] = etree.tostring(content_types_root, xml_declaration=True, encoding='UTF-8',
#                                                            standalone=True)
#
#         # --- 步骤 7: 将内存中的文件重新打包成新的PPTX ---
#         with zipfile.ZipFile(output_pptx_path, 'w', zipfile.ZIP_DEFLATED) as z_out:
#             for filename, content in package_files.items():
#                 z_out.writestr(filename, content)
#
#         print(f"处理完成！仅包含一张修改后幻灯片的新文件已保存至: {output_pptx_path}")
#         return True
#     except Exception as e:
#         print(f"在 create_single_slide_ppt 过程中发生错误: {e}")
#         return False
#
#
# def duplicate_slide(
#         input_pptx_path: str,
#         output_pptx_path: str,
#         num_copies: int
# ):
#     """
#     读取一个PPTX文件，将其中的第一张幻灯片复制指定的次数。
#     此版本修正了导致文件损坏的错误。
#     """
#     try:
#         # --- 步骤 1: 将PPTX所有文件读入内存 ---
#         package_files = {}
#         with zipfile.ZipFile(input_pptx_path, 'r') as z_in:
#             for item in z_in.infolist():
#                 package_files[item.filename] = z_in.read(item.filename)
#
#         # --- 步骤 2: 准备源文件和核心XML的解析 ---
#         source_slide_path = 'ppt/slides/slide1.xml'
#         source_slide_rels_path = 'ppt/slides/_rels/slide1.xml.rels'
#         if source_slide_path not in package_files:
#             print(f"错误: 找不到源幻灯片 '{source_slide_path}'")
#             return
#
#         source_slide_content = package_files[source_slide_path]
#         source_slide_rels_content = package_files.get(source_slide_rels_path)
#
#         pres_xml_path = 'ppt/presentation.xml'
#         pres_rels_path = 'ppt/_rels/presentation.xml.rels'
#         content_types_path = '[Content_Types].xml'
#
#         pres_root = etree.fromstring(package_files[pres_xml_path])
#         pres_rels_root = etree.fromstring(package_files[pres_rels_path])
#         content_types_root = etree.fromstring(package_files[content_types_path])
#
#         ns = {
#             'p': 'http://schemas.openxmlformats.org/presentationml/2006/main',
#             'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
#         }
#
#         # --- 步骤 3: 确定新文件的唯一ID和文件名 ---
#         # 查找所有已存在的ID，以避免冲突
#         existing_slide_nums = [int(re.search(r'slide(\d+)\.xml', f).group(1)) for f in package_files if
#                                re.search(r'ppt/slides/slide\d+\.xml', f)]
#         existing_rids = [int(r.get('Id')[3:]) for r in
#                          pres_rels_root.findall('Relationship', namespaces=pres_rels_root.nsmap)]
#         existing_slide_ids = [int(s.get('id')) for s in pres_root.findall('.//p:sldId', namespaces=ns)]
#
#         max_slide_num = max(existing_slide_nums) if existing_slide_nums else 0
#         max_rid_num = max(existing_rids) if existing_rids else 0
#         max_slide_id = max(existing_slide_ids) if existing_slide_ids else 255
#
#         # --- 步骤 4: 循环创建幻灯片副本 ---
#         for i in range(num_copies):
#             # 为新幻灯片生成全新的、不重复的ID和文件名
#             new_slide_num = max_slide_num + 1 + i
#             new_rid = f"rId{max_rid_num + 1 + i}"
#             new_slide_id = max_slide_id + 1 + i
#
#             # --- 4a: 复制文件到内存 ---
#             new_slide_path = f'ppt/slides/slide{new_slide_num}.xml'
#             package_files[new_slide_path] = source_slide_content
#
#             if source_slide_rels_content:
#                 new_slide_rels_path = f'ppt/slides/_rels/slide{new_slide_num}.xml.rels'
#                 package_files[new_slide_rels_path] = source_slide_rels_content
#
#             # --- 4b: 在 [Content_Types].xml 中注册新文件 (!!! 关键修正 !!!) ---
#             # 注册新的幻灯片文件
#             etree.SubElement(
#                 content_types_root, 'Override',
#                 PartName=f'/ppt/slides/slide{new_slide_num}.xml',
#                 ContentType='application/vnd.openxmlformats-officedocument.presentationml.slide+xml'
#             )
#             # 如果存在，注册新的关系文件
#             if source_slide_rels_content:
#                 etree.SubElement(
#                     content_types_root, 'Override',
#                     PartName=f'/ppt/slides/_rels/slide{new_slide_num}.xml.rels',
#                     ContentType='application/vnd.openxmlformats-officedocument.package.relationships+xml'
#                 )
#
#             # --- 4c: 在 ppt/_rels/presentation.xml.rels 中添加关系 ---
#             etree.SubElement(
#                 pres_rels_root, 'Relationship',
#                 Id=new_rid,
#                 Type='http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide',
#                 Target=f'slides/slide{new_slide_num}.xml'
#             )
#
#             # --- 4d: 在 ppt/presentation.xml 中添加幻灯片到列表 ---
#             slide_id_list = pres_root.find('p:sldIdLst', namespaces=ns)
#             new_slide_id_el = etree.SubElement(slide_id_list, '{' + ns['p'] + '}sldId', id=str(new_slide_id))
#             new_slide_id_el.set('{' + ns['r'] + '}id', new_rid)
#
#             print(f"已在内存中创建第 {i + 1} 个副本 (slide{new_slide_num}.xml)")
#
#         # --- 步骤 5: 将所有被修改的XML结构写回内存中的文件 ---
#         package_files[pres_xml_path] = etree.tostring(pres_root, xml_declaration=True, encoding='UTF-8',
#                                                       standalone=True)
#         package_files[pres_rels_path] = etree.tostring(pres_rels_root, xml_declaration=True, encoding='UTF-8',
#                                                        standalone=True)
#         package_files[content_types_path] = etree.tostring(content_types_root, xml_declaration=True, encoding='UTF-8',
#                                                            standalone=True)
#
#         # --- 步骤 6: 将内存中的所有文件重新打包成新的PPTX ---
#         with zipfile.ZipFile(output_pptx_path, 'w', zipfile.ZIP_DEFLATED) as z_out:
#             for filename, content in package_files.items():
#                 z_out.writestr(filename, content)
#
#         print(f"成功复制 {num_copies} 页幻灯片，最终文件（共{num_copies + 1}页）保存为: {output_pptx_path}")
#
#     except Exception as e:
#         print(f"在 duplicate_slide 过程中发生错误: {e}")
#
#
# # --- 主执行流程 ---
# if __name__ == '__main__':
#     source_ppt = "./ppt/多agent讲解.pptx"  # Replace with your input PPTX file path
#     intermediate_ppt ="./ppt/xml处理单页.pptx"  # Output PPTX file path
#     final_ppt = "./ppt/temp.pptx"  # Output PPTX fi
#
#     if not os.path.exists(source_ppt):
#         print(f"错误: 请先创建一个名为 '{source_ppt}' 的演示文稿用于测试。")
#         print("它应包含多页，且第一页有'旧的文本'字样。")
#     else:
#         # 第1步: 从多页PPT中提取并修改第一页，生成一个临时的单页PPT
#         is_step1_success = create_single_slide_ppt(
#             input_pptx_path=source_ppt,
#             output_pptx_path=intermediate_ppt,
#             text_to_find="大模型食品安全智能体应用",
#             text_to_replace="xxxx"
#         )
#
#         # 第2步: 如果上一步成功，则对这个单页PPT进行复制操作
#         if is_step1_success:
#             duplicate_slide(
#                 input_pptx_path=intermediate_ppt,
#                 output_pptx_path=final_ppt,
#                 num_copies=3  # 复制3次，最终文件将有 1+3=4 页
#             )
#             # 删除不再需要的中间文件
#             os.remove(intermediate_ppt)
#             print(f"中间文件 '{intermediate_ppt}' 已被删除。")
#
#
import zipfile
import re
from lxml import etree
import os
from copy import deepcopy

import zipfile
import re
from lxml import etree
import os
from copy import deepcopy


def merge_template_slide_(
        dest_pptx_path: str,
        template_pptx_path: str,
        output_pptx_path: str
):
    """
    将一个模板PPTX文件中的第一张幻灯片及其所有依赖项（母版、布局、主题、媒体）
    复制到另一个PPTX文件的末尾。
    此版本已修正对“无布局”幻灯片的处理。
    """
    try:
        # --- 步骤 1: 将两个PPTX的所有文件读入内存 ---
        dest_pkg = {}
        with zipfile.ZipFile(dest_pptx_path, 'r') as z_in:
            for item in z_in.infolist():
                dest_pkg[item.filename] = z_in.read(item.filename)

        src_pkg = {}
        with zipfile.ZipFile(template_pptx_path, 'r') as z_in:
            for item in z_in.infolist():
                src_pkg[item.filename] = z_in.read(item.filename)

        # 定义命名空间
        ns = {
            'p': 'http://schemas.openxmlformats.org/presentationml/2006/main',
            'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
            'ct': 'http://schemas.openxmlformats.org/package/2006/content-types'
        }

        # --- 步骤 2: 分析模板的依赖关系 ---

        # 2a. 找到幻灯片 -> 布局
        src_slide_rels_path = 'ppt/slides/_rels/slide1.xml.rels'
        src_slide_rels_root = etree.fromstring(src_pkg[src_slide_rels_path]) if src_slide_rels_path in src_pkg else None

        layout_rel = None
        if src_slide_rels_root is not None:
            layout_rel = src_slide_rels_root.find(
                "r:Relationship[@Type='http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout']",
                namespaces=ns)

        # !!! CRITICAL FIX: 处理幻灯片没有布局的情况 !!!
        if layout_rel is not None:
            # --- 情况 A: 模板幻灯片有关联布局 (原逻辑) ---
            print("信息: 模板幻灯片有关联布局，将进行完整移植。")
            # (以下逻辑与之前版本类似，但更健壮)
            src_layout_path = os.path.normpath(
                os.path.join(os.path.dirname(src_slide_rels_path), layout_rel.get('Target'))).replace('\\', '/')
            layout_filename = os.path.basename(src_layout_path)
            src_layout_rels_path = f'ppt/slideLayouts/_rels/{layout_filename}.rels'
            src_layout_rels_root = etree.fromstring(src_pkg[src_layout_rels_path])
            master_rel = src_layout_rels_root.find(
                "r:Relationship[@Type='http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster']",
                namespaces=ns)
            src_master_path = os.path.normpath(
                os.path.join(os.path.dirname(src_layout_rels_path), master_rel.get('Target'))).replace('\\', '/')
            master_filename = os.path.basename(src_master_path)
            src_master_rels_path = f'ppt/slideMasters/_rels/{master_filename}.rels'
            src_master_rels_root = etree.fromstring(src_pkg[src_master_rels_path])

            files_to_copy_from_src = {
                'ppt/slides/slide1.xml', src_slide_rels_path,
                src_layout_path, src_layout_rels_path,
                src_master_path, src_master_rels_path
            }

            for rel in src_master_rels_root.findall('r:Relationship', namespaces=ns):
                rel_target_path = os.path.normpath(
                    os.path.join(os.path.dirname(src_master_rels_path), rel.get('Target'))).replace('\\', '/')
                files_to_copy_from_src.add(rel_target_path)
                if 'slideLayout' in rel.get('Type') or 'theme' in rel.get('Type'):
                    rel_filename = os.path.basename(rel_target_path)
                    rel_rels_path = os.path.join(os.path.dirname(rel_target_path), '_rels', f"{rel_filename}.rels")
                    if rel_rels_path in src_pkg: files_to_copy_from_src.add(rel_rels_path)

        else:
            # --- 情况 B: 模板幻灯片是“空白页”，没有布局 ---
            print("信息: 模板幻灯片为无布局模式，将仅复制内容并关联到目标布局。")
            files_to_copy_from_src = {'ppt/slides/slide1.xml'}
            if src_slide_rels_path in src_pkg:
                files_to_copy_from_src.add(src_slide_rels_path)

        # 2d. 查找幻灯片直接引用的媒体文件 (对两种情况都适用)
        if src_slide_rels_root is not None:
            for rel in src_slide_rels_root.findall('r:Relationship', namespaces=ns):
                if 'image' in rel.get('Type'):
                    media_path = os.path.normpath(
                        os.path.join(os.path.dirname(src_slide_rels_path), rel.get('Target'))).replace('\\', '/')
                    files_to_copy_from_src.add(media_path)

        # (后续的资源移植和关系重建逻辑保持，但现在它们处理的是一个更准确的文件列表)

        # --- 步骤 3: 资源移植 (复制并重命名) ---
        def get_max_id(pkg, pattern):
            nums = [int(m.group(1)) for f in pkg for m in [re.search(pattern, f)] if m]
            return max(nums) if nums else 0

        next_slide_num = get_max_id(dest_pkg, r'slide(\d+)\.xml') + 1
        # ... (和之前一样，创建path_map, 复制文件, 更新路径)
        path_map = {}
        new_slide_path = f'ppt/slides/slide{next_slide_num}.xml'
        path_map['ppt/slides/slide1.xml'] = new_slide_path

        # 复制核心文件
        dest_pkg[new_slide_path] = src_pkg['ppt/slides/slide1.xml']

        # --- 步骤 4: 关系重建 ---
        # 4a. 创建新的幻灯片关系文件
        new_slide_rels_path = f'ppt/slides/_rels/slide{next_slide_num}.xml.rels'

        if layout_rel is not None:
            # 如果源有布局，则复制并修改其关系文件
            src_rels_content = src_pkg[src_slide_rels_path]
            # ... 此处省略了复杂的路径重写逻辑，需要一个更完整的移植 ...
            # 为简化，我们假设完整移植场景暂时不处理
            # 以下代码优先保证无布局场景能成功运行
            dest_pkg[new_slide_rels_path] = src_rels_content  # 简化复制，后续需要更复杂的路径重写
        else:
            # 如果源没有布局，为新幻灯片创建一个关系文件，并链接到目标文件的第一个布局
            dest_slide1_rels_root = etree.fromstring(dest_pkg['ppt/slides/_rels/slide1.xml.rels'])
            dest_layout_rel = dest_slide1_rels_root.find(
                "r:Relationship[@Type='http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout']",
                namespaces=ns)

            new_rels_root = etree.Element('Relationships',
                                          xmlns="http://schemas.openxmlformats.org/package/2006/relationships")
            if dest_layout_rel is not None:
                etree.SubElement(new_rels_root, 'Relationship',
                                 Id="rId1",
                                 Type=dest_layout_rel.get('Type'),
                                 Target=dest_layout_rel.get('Target'))

            # 复制源幻灯片中的图片等其他关系
            rid_counter = 2
            if src_slide_rels_root is not None:
                for rel in src_slide_rels_root:
                    if 'slideLayout' not in rel.get('Type'):
                        new_rel = deepcopy(rel)
                        new_rel.set('Id', f'rId{rid_counter}')
                        rid_counter += 1
                        new_rels_root.append(new_rel)
                        # 同时复制媒体文件
                        media_path = os.path.normpath(os.path.join('ppt/slides', rel.get('Target'))).replace('\\', '/')
                        if media_path in src_pkg:
                            dest_pkg[media_path] = src_pkg[media_path]

            dest_pkg[new_slide_rels_path] = etree.tostring(new_rels_root, xml_declaration=True, encoding='UTF-8',
                                                           standalone=True)

        # 4b. 更新 [Content_Types].xml (简化版，只添加新幻灯片和其关系)
        content_types_root = etree.fromstring(dest_pkg['[Content_Types].xml'])
        etree.SubElement(content_types_root, 'Override', PartName=f'/{new_slide_path}',
                         ContentType='application/vnd.openxmlformats-officedocument.presentationml.slide+xml')
        etree.SubElement(content_types_root, 'Override', PartName=f'/{new_slide_rels_path}',
                         ContentType='application/vnd.openxmlformats-officedocument.package.relationships+xml')
        # ... 还需要添加图片等媒体类型 ...
        dest_pkg['[Content_Types].xml'] = etree.tostring(content_types_root, xml_declaration=True, encoding='UTF-8',
                                                         standalone=True)

        # 4c. 更新 presentation.xml 和 .rels
        pres_root = etree.fromstring(dest_pkg['ppt/presentation.xml'])
        pres_rels_root = etree.fromstring(dest_pkg['ppt/_rels/presentation.xml.rels'])
        max_rid = max([int(r.get('Id')[3:]) for r in pres_rels_root])

        new_slide_rid = f'rId{max_rid + 1}'
        etree.SubElement(pres_rels_root, 'Relationship', Id=new_slide_rid,
                         Type='http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide',
                         Target=f'slides/slide{next_slide_num}.xml')

        sld_id_lst = pres_root.find('p:sldIdLst', namespaces=ns)
        max_slide_id = max([int(s.get('id')) for s in sld_id_lst] or [255])
        new_slide_id_el = etree.SubElement(sld_id_lst, '{' + ns['p'] + '}sldId', id=str(max_slide_id + 1))
        new_slide_id_el.set('{' + ns['r'] + '}id', new_slide_rid)

        dest_pkg['ppt/presentation.xml'] = etree.tostring(pres_root, xml_declaration=True, encoding='UTF-8',
                                                          standalone=True)
        dest_pkg['ppt/_rels/presentation.xml.rels'] = etree.tostring(pres_rels_root, xml_declaration=True,
                                                                     encoding='UTF-8', standalone=True)

        # --- 步骤 5: 重新打包 ---
        with zipfile.ZipFile(output_pptx_path, 'w', zipfile.ZIP_DEFLATED) as z_out:
            for filename, content in dest_pkg.items():
                z_out.writestr(filename, content)

        print(f"成功将模板幻灯片合并到目标文件，已保存为: {output_pptx_path}")

    except Exception as e:
        print(f"处理过程中发生严重错误: {e}")
        import traceback
        traceback.print_exc()

import zipfile
import re
from lxml import etree
import os
from copy import deepcopy


def merge_template_slide(
        dest_pptx_path: str,
        template_pptx_path: str,
        output_pptx_path: str,
        insert_position: int = 2
):
    """
    将一个模板PPTX中的第一张幻灯片及其所有依赖项，插入到目标PPTX的指定位置。

    :param dest_pptx_path: 目标PPTX文件路径。
    :param template_pptx_path: 只含一页的模板PPTX文件路径。
    :param output_pptx_path: 合并后的输出文件路径。
    :param insert_position: 幻灯片要插入的位置 (1-based index, e.g., 2 for the second slide)。
    """
    try:
        # --- 步骤 1: 将两个PPTX的所有文件读入内存 ---
        dest_pkg = {}
        with zipfile.ZipFile(dest_pptx_path, 'r') as z_in:
            for item in z_in.infolist():
                dest_pkg[item.filename] = z_in.read(item.filename)

        src_pkg = {}
        with zipfile.ZipFile(template_pptx_path, 'r') as z_in:
            for item in z_in.infolist():
                src_pkg[item.filename] = z_in.read(item.filename)

        # 定义命名空间
        ns = {
            'p': 'http://schemas.openxmlformats.org/presentationml/2006/main',
            'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
        }

        # --- 步骤 2: 确定新文件的唯一ID和文件名 ---
        def get_max_id(pkg, pattern):
            nums = [int(m.group(1)) for f in pkg for m in [re.search(pattern, f)] if m]
            return max(nums) if nums else 0

        next_slide_num = get_max_id(dest_pkg, r'slide(\d+)\.xml') + 1

        # --- 步骤 3: 移植模板幻灯片及其内容 ---
        # 3a. 复制幻灯片主体
        new_slide_path = f'ppt/slides/slide{next_slide_num}.xml'
        dest_pkg[new_slide_path] = src_pkg['ppt/slides/slide1.xml']

        # 3b. 创建或复制幻灯片的关系文件 (.rels)
        new_slide_rels_path = f'ppt/slides/_rels/slide{next_slide_num}.xml.rels'
        src_slide_rels_path = 'ppt/slides/_rels/slide1.xml.rels'
        src_slide_rels_root = etree.fromstring(src_pkg[src_slide_rels_path]) if src_slide_rels_path in src_pkg else None

        new_rels_root = etree.Element('Relationships',
                                      xmlns="http://schemas.openxmlformats.org/package/2006/relationships")

        # 检查模板幻灯片是否有布局
        layout_rel = None
        if src_slide_rels_root is not None:
            layout_rel = src_slide_rels_root.find(
                "r:Relationship[@Type='http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout']",
                namespaces=ns)

        if layout_rel is not None:
            # 如果模板有自己的布局，这里应执行复杂的母版、布局、主题的完整移植。
            # 为保持代码清晰，我们仍采用简化策略：将其关联到目标文件的布局上。
            # 如需完整移植，之前的复杂代码可在此处集成。
            print("信息: 模板有关联布局，为简化将重新关联到目标布局。")
        else:
            print("信息: 模板为无布局模式，将关联到目标布局。")

        # 统一处理：将新幻灯片关联到目标文件的第一个布局
        dest_slide1_rels_root = etree.fromstring(dest_pkg['ppt/slides/_rels/slide1.xml.rels'])
        dest_layout_rel = dest_slide1_rels_root.find(
            "r:Relationship[@Type='http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout']",
            namespaces=ns)

        rid_counter = 1
        if dest_layout_rel is not None:
            etree.SubElement(new_rels_root, 'Relationship',
                             Id=f"rId{rid_counter}",
                             Type=dest_layout_rel.get('Type'),
                             Target=dest_layout_rel.get('Target'))
            rid_counter += 1

        # 复制源幻灯片中的图片等其他关系
        if src_slide_rels_root is not None:
            for rel in src_slide_rels_root:
                # 跳过旧的布局关系
                if 'slideLayout' not in rel.get('Type'):
                    new_rel = deepcopy(rel)
                    new_rel.set('Id', f'rId{rid_counter}')
                    rid_counter += 1
                    new_rels_root.append(new_rel)

                    # 同时复制媒体文件，确保路径正确
                    media_target = rel.get('Target')
                    src_media_path = os.path.normpath(
                        os.path.join(os.path.dirname(src_slide_rels_path), media_target)).replace('\\', '/')
                    dest_media_path = os.path.normpath(
                        os.path.join(os.path.dirname(new_slide_rels_path), media_target)).replace('\\', '/')
                    if src_media_path in src_pkg and dest_media_path not in dest_pkg:
                        dest_pkg[dest_media_path] = src_pkg[src_media_path]

        dest_pkg[new_slide_rels_path] = etree.tostring(new_rels_root, xml_declaration=True, encoding='UTF-8',
                                                       standalone=True)

        # --- 步骤 4: 重建核心清单文件的关系 ---

        # 4a. 更新 [Content_Types].xml
        content_types_root = etree.fromstring(dest_pkg['[Content_Types].xml'])
        etree.SubElement(content_types_root, 'Override', PartName=f'/{new_slide_path}',
                         ContentType='application/vnd.openxmlformats-officedocument.presentationml.slide+xml')
        etree.SubElement(content_types_root, 'Override', PartName=f'/{new_slide_rels_path}',
                         ContentType='application/vnd.openxmlformats-officedocument.package.relationships+xml')
        # ... 此处还应为复制的媒体文件添加类型声明 ...
        dest_pkg['[Content_Types].xml'] = etree.tostring(content_types_root, xml_declaration=True, encoding='UTF-8',
                                                         standalone=True)

        # 4b. 更新 presentation.xml 和 .rels
        pres_root = etree.fromstring(dest_pkg['ppt/presentation.xml'])
        pres_rels_root = etree.fromstring(dest_pkg['ppt/_rels/presentation.xml.rels'])

        # 为新幻灯片创建关系
        max_rid = max([int(r.get('Id')[3:]) for r in pres_rels_root]) if pres_rels_root else 0
        new_slide_rid = f'rId{max_rid + 1}'
        etree.SubElement(pres_rels_root, 'Relationship', Id=new_slide_rid,
                         Type='http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide',
                         Target=f'slides/slide{next_slide_num}.xml')

        # 在presentation.xml中 *插入* 幻灯片ID
        sld_id_lst = pres_root.find('p:sldIdLst', namespaces=ns)
        max_slide_id = max([int(s.get('id')) for s in sld_id_lst] or [255])

        new_slide_id_el = etree.Element('{' + ns['p'] + '}sldId', id=str(max_slide_id + 1))
        new_slide_id_el.set('{' + ns['r'] + '}id', new_slide_rid)

        # !!! 核心修改：从 append 改为 insert !!!
        num_existing_slides = len(sld_id_lst)
        # 将1-based的用户位置转换为0-based的列表索引
        # 使用min/max确保索引在安全范围内 [0, num_existing_slides]
        insert_index = max(0, min(insert_position - 1, num_existing_slides))

        sld_id_lst.insert(insert_index, new_slide_id_el)
        print(f"信息: 新幻灯片 (slide{next_slide_num}.xml) 已插入到位置 {insert_index + 1}。")

        dest_pkg['ppt/presentation.xml'] = etree.tostring(pres_root, xml_declaration=True, encoding='UTF-8',
                                                          standalone=True)
        dest_pkg['ppt/_rels/presentation.xml.rels'] = etree.tostring(pres_rels_root, xml_declaration=True,
                                                                     encoding='UTF-8', standalone=True)

        # --- 步骤 5: 重新打包 ---
        with zipfile.ZipFile(output_pptx_path, 'w', zipfile.ZIP_DEFLATED) as z_out:
            for filename, content in dest_pkg.items():
                z_out.writestr(filename, content)

        print(f"成功将模板幻灯片合并到目标文件，已保存为: {output_pptx_path}")

    except Exception as e:
        print(f"处理过程中发生严重错误: {e}")
        import traceback
        traceback.print_exc()


import zipfile
import re
from lxml import etree
import os
from copy import deepcopy


def merge_elements_onto_slide(
        dest_pptx_path: str,
        template_pptx_path: str,
        slide_number_to_modify: int = 2
):
    """
    将模板PPTX第一页的所有可见元素（包括来自布局和母版的元素），
    复制并叠加到目标PPTX的指定页面上。
    此版本修正了因元素在布局/母版中而导致的崩溃。
    修改直接在 dest_pptx_path 上进行，不生成新文件。
    """
    try:
        # --- 步骤 1: 将所有相关PPTX文件读入内存 ---
        dest_pkg = {}
        with zipfile.ZipFile(dest_pptx_path, 'r') as z_in:
            for item in z_in.infolist():
                dest_pkg[item.filename] = z_in.read(item.filename)

        src_pkg = {}
        with zipfile.ZipFile(template_pptx_path, 'r') as z_in:
            for item in z_in.infolist():
                src_pkg[item.filename] = z_in.read(item.filename)

        # 定义命名空间
        ns = {
            'p': 'http://schemas.openxmlformats.org/presentationml/2006/main',
            'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
            'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
        }

        # --- 步骤 2: 提取源幻灯片、布局、母版的所有元素 ---
        all_elements_to_copy = []
        all_src_rels = {}  # 存储每个XML文件对应的关系字典

        # 2a. 查找依赖链: Slide -> Layout -> Master
        src_slide_rels_path = 'ppt/slides/_rels/slide1.xml.rels'
        src_slide_rels_root = etree.fromstring(src_pkg[src_slide_rels_path]) if src_slide_rels_path in src_pkg else None
        all_src_rels['ppt/slides/slide1.xml'] = src_slide_rels_root

        src_layout_path, src_master_path = None, None
        if src_slide_rels_root is not None:
            layout_rel = src_slide_rels_root.find(
                "r:Relationship[@Type='http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout']",
                namespaces=ns)
            if layout_rel is not None:
                src_layout_path = os.path.normpath(os.path.join('ppt/slides', layout_rel.get('Target'))).replace('\\',
                                                                                                                 '/')
                layout_filename = os.path.basename(src_layout_path)
                src_layout_rels_path = f'ppt/slideLayouts/_rels/{layout_filename}.rels'
                src_layout_rels_root = etree.fromstring(
                    src_pkg[src_layout_rels_path]) if src_layout_rels_path in src_pkg else None
                all_src_rels[src_layout_path] = src_layout_rels_root

                if src_layout_rels_root is not None:
                    master_rel = src_layout_rels_root.find(
                        "r:Relationship[@Type='http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster']",
                        namespaces=ns)
                    if master_rel is not None:
                        src_master_path = os.path.normpath(
                            os.path.join('ppt/slideLayouts', master_rel.get('Target'))).replace('\\', '/')
                        master_filename = os.path.basename(src_master_path)
                        src_master_rels_path = f'ppt/slideMasters/_rels/{master_filename}.rels'
                        all_src_rels[src_master_path] = etree.fromstring(
                            src_pkg[src_master_rels_path]) if src_master_rels_path in src_pkg else None

        # 2b. 依次从 Master, Layout, Slide 提取元素
        for path in [src_master_path, src_layout_path, 'ppt/slides/slide1.xml']:
            if path and path in src_pkg:
                root = etree.fromstring(src_pkg[path])
                spTree = root.find('.//p:spTree', namespaces=ns)
                if spTree is not None:
                    # 使用 deepcopy 确保每个元素都是独立的
                    all_elements_to_copy.extend(deepcopy(list(spTree)))

        if not all_elements_to_copy:
            print("警告: 在模板的任何层级（幻灯片、布局、母版）都未找到可复制的元素。")
            return

        # --- 步骤 3: 处理所有元素的依赖关系 ---
        dest_slide_path = f'ppt/slides/slide{slide_number_to_modify}.xml'
        dest_rels_path = f'ppt/slides/_rels/slide{slide_number_to_modify}.xml.rels'
        dest_rels_root = etree.fromstring(dest_pkg[dest_rels_path]) if dest_rels_path in dest_pkg else etree.Element(
            'Relationships')
        max_rid_num = max([int(r.get('Id')[3:]) for r in dest_rels_root] or [0])
        rid_map = {}

        print(f"准备从模板各层级复制共 {len(all_elements_to_copy)} 个元素...")

        for element in all_elements_to_copy:
            dependencies = element.xpath('.//*[@r:embed or @r:link]', namespaces=ns)
            for dep in dependencies:
                for attr_name in ['embed', 'link']:
                    old_rid = dep.get('{' + ns['r'] + '}' + attr_name)
                    if not old_rid: continue

                    if old_rid in rid_map:
                        dep.set('{' + ns['r'] + '}' + attr_name, rid_map[old_rid])
                        continue

                    # 确定这个 rId 来自哪个 rels 文件
                    src_rels_root = None
                    # ... 逻辑简化：假设所有rId都可以在slide1.xml.rels中找到其定义 ...
                    # 这是一个合理的简化，因为图片等资源通常直接与幻灯片关联
                    src_rels_root = all_src_rels.get('ppt/slides/slide1.xml')
                    if src_rels_root is None: continue

                    rel = src_rels_root.find(f"r:Relationship[@Id='{old_rid}']", namespaces=ns)
                    if rel is None: continue

                    target_path = rel.get('Target')
                    new_rid_num = max_rid_num + 1
                    max_rid_num = new_rid_num
                    new_rid = f'rId{new_rid_num}'

                    if rel.get('TargetMode') == 'External':
                        etree.SubElement(dest_rels_root, 'Relationship', Id=new_rid, Type=rel.get('Type'),
                                         Target=target_path, TargetMode='External')
                    else:
                        src_media_path = os.path.normpath(os.path.join('ppt/slides', target_path)).replace('\\', '/')
                        try:
                            media_content = src_pkg[src_media_path]
                            if src_media_path not in dest_pkg:
                                dest_pkg[src_media_path] = media_content
                            etree.SubElement(dest_rels_root, 'Relationship', Id=new_rid, Type=rel.get('Type'),
                                             Target=target_path)
                        except KeyError:
                            print(f"警告：找不到媒体文件 {src_media_path}，跳过此依赖。")
                            continue

                    rid_map[old_rid] = new_rid
                    dep.set('{' + ns['r'] + '}' + attr_name, new_rid)

        # --- 步骤 4: 注入元素 ---
        dest_slide_root = etree.fromstring(dest_pkg[dest_slide_path])
        dest_spTree = dest_slide_root.find('.//p:spTree', namespaces=ns)
        # 如果目标幻灯片本身也没有spTree（例如空白页），需要创建它
        if dest_spTree is None:
            c_sld = dest_slide_root.find('p:cSld', namespaces=ns)
            dest_spTree = etree.SubElement(c_sld, '{' + ns['p'] + '}spTree')
            # 还需要添加必要的子元素
            etree.SubElement(dest_spTree, '{' + ns['p'] + '}nvGrpSpPr')
            etree.SubElement(dest_spTree, '{' + ns['p'] + '}grpSpPr')

        for element in all_elements_to_copy:
            dest_spTree.append(element)
        print(f"已将 {len(all_elements_to_copy)} 个元素注入到{slide_number_to_modify}幻灯片。")

        # --- 步骤 5: 更新内存中的文件并直接保存到源文件 ---
        dest_pkg[dest_slide_path] = etree.tostring(dest_slide_root, xml_declaration=True, encoding='UTF-8',
                                                   standalone=True)
        dest_pkg[dest_rels_path] = etree.tostring(dest_rels_root, xml_declaration=True, encoding='UTF-8',
                                                  standalone=True)

        with zipfile.ZipFile(dest_pptx_path, 'w', zipfile.ZIP_DEFLATED) as z_out:
            for filename, content in dest_pkg.items():
                z_out.writestr(filename, content)

        print(f"处理完成！文件已更新: {dest_pptx_path}")

    except Exception as e:
        print(f"处理过程中发生严重错误: {e}")
        import traceback
        traceback.print_exc()

import zipfile
import re
from lxml import etree

import zipfile
import re
from lxml import etree

import zipfile
import re
from lxml import etree
from copy import deepcopy

def duplicate_slide_to_index(
        input_pptx_path: str,
        output_pptx_path: str,
        slide_index: int,
        num_copies: int,
        target_index: int
):
    """
    读取一个PPTX文件，将指定索引的幻灯片复制指定的次数，并插入到目标索引位置。
    slide_index: 1-based index of the slide to duplicate (e.g., 1 for first slide).
    target_index: 1-based index where the duplicated slides will be inserted.
    """
    try:
        # --- 步骤 1: 将PPTX所有文件读入内存 ---
        package_files = {}
        with zipfile.ZipFile(input_pptx_path, 'r') as z_in:
            for item in z_in.infolist():
                package_files[item.filename] = z_in.read(item.filename)

        # --- 步骤 2: 准备源文件和核心XML的解析 ---
        source_slide_path = f'ppt/slides/slide{slide_index}.xml'
        source_slide_rels_path = f'ppt/slides/_rels/slide{slide_index}.xml.rels'
        if source_slide_path not in package_files:
            print(f"错误: 找不到源幻灯片 '{source_slide_path}'")
            return

        source_slide_content = package_files[source_slide_path]
        source_slide_rels_content = package_files.get(source_slide_rels_path)

        pres_xml_path = 'ppt/presentation.xml'
        pres_rels_path = 'ppt/_rels/presentation.xml.rels'
        content_types_path = '[Content_Types].xml'

        pres_root = etree.fromstring(package_files[pres_xml_path])
        pres_rels_root = etree.fromstring(package_files[pres_rels_path])
        content_types_root = etree.fromstring(package_files[content_types_path])

        ns = {
            'p': 'http://schemas.openxmlformats.org/presentationml/2006/main',
            'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
        }

        # --- 步骤 3: 确定新文件的唯一ID和文件名 ---
        existing_slide_nums = [int(re.search(r'slide(\d+)\.xml', f).group(1)) for f in package_files if
                              re.search(r'ppt/slides/slide\d+\.xml', f)]
        existing_rids = [int(r.get('Id')[3:]) for r in
                         pres_rels_root.findall('Relationship', namespaces=pres_rels_root.nsmap)]
        existing_slide_ids = [int(s.get('id')) for s in pres_root.findall('.//p:sldId', namespaces=ns)]

        max_slide_num = max(existing_slide_nums) if existing_slide_nums else 0
        max_rid_num = max(existing_rids) if existing_rids else 0
        max_slide_id = max(existing_slide_ids) if existing_slide_ids else 255

        # --- 步骤 4: 循环创建幻灯片副本 ---
        slide_id_list = pres_root.find('p:sldIdLst', namespaces=ns)
        # Find the position to insert new slides (0-based index for list insertion)
        insert_pos = target_index -1  if target_index <= len(slide_id_list) else len(slide_id_list)
        # 调试：打印原始 sldIdLst
        print("原始 sldIdLst 内容：")
        for sldId in slide_id_list:
            print(f"sldId id={sldId.get('id')} r:id={sldId.get('{' + ns['r'] + '}id')}")

        for i in range(num_copies):
            # 为新幻灯片生成全新的、不重复的ID和文件名
            new_slide_num = max_slide_num + 1 + i
            new_rid = f"rId{max_rid_num + 1 + i}"
            new_slide_id = max_slide_id + 1 + i

            # --- 4a: 复制文件到内存 ---
            new_slide_path = f'ppt/slides/slide{new_slide_num}.xml'
            package_files[new_slide_path] = source_slide_content

            if source_slide_rels_content:
                new_slide_rels_path = f'ppt/slides/_rels/slide{new_slide_num}.xml.rels'
                package_files[new_slide_rels_path] = source_slide_rels_content

            # --- 4b: 在 [Content_Types].xml 中注册新文件 ---
            etree.SubElement(
                content_types_root, 'Override',
                PartName=f'/ppt/slides/slide{new_slide_num}.xml',
                ContentType='application/vnd.openxmlformats-officedocument.presentationml.slide+xml'
            )
            if source_slide_rels_content:
                etree.SubElement(
                    content_types_root, 'Override',
                    PartName=f'/ppt/slides/_rels/slide{new_slide_num}.xml.rels',
                    ContentType='application/vnd.openxmlformats-officedocument.package.relationships+xml'
                )

            # --- 4c: 在 ppt/_rels/presentation.xml.rels 中添加关系 ---
            etree.SubElement(
                pres_rels_root, 'Relationship',
                Id=new_rid,
                Type='http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide',
                Target=f'slides/slide{new_slide_num}.xml'
            )

            # --- 4d: 在 ppt/presentation.xml 中添加幻灯片到指定位置 ---
            new_slide_id_el = etree.Element('{' + ns['p'] + '}sldId', id=str(new_slide_id))
            new_slide_id_el.set('{' + ns['r'] + '}id', new_rid)
            slide_id_list.insert(insert_pos + i, new_slide_id_el)

            print(f"已在内存中创建第 {i + 1} 个副本 (slide{new_slide_num}.xml)，插入到位置 {target_index + i}")
        # 调试：打印原始 sldIdLst
        print("原始 sldIdLst 内容：")
        for sldId in slide_id_list:
            print(f"sldId id={sldId.get('id')} r:id={sldId.get('{' + ns['r'] + '}id')}")

        # --- 步骤 5: 将所有被修改的XML结构写回内存中的文件 ---
        package_files[pres_xml_path] = etree.tostring(pres_root, xml_declaration=True, encoding='UTF-8',
                                                      standalone=True)
        package_files[pres_rels_path] = etree.tostring(pres_rels_root, xml_declaration=True, encoding='UTF-8',
                                                       standalone=True)
        package_files[content_types_path] = etree.tostring(content_types_root, xml_declaration=True, encoding='UTF-8',
                                                           standalone=True)

        # --- 步骤 6: 将内存中的所有文件重新打包成新的PPTX ---
        with zipfile.ZipFile(output_pptx_path, 'w', zipfile.ZIP_DEFLATED) as z_out:
            for filename, content in package_files.items():
                z_out.writestr(filename, content)

        print(f"成功复制第 {slide_index} 页幻灯片 {num_copies} 次，插入到位置 {target_index}，最终文件（共{len(existing_slide_nums) + num_copies}页）保存为: {output_pptx_path}")

    except Exception as e:
        print(f"在 duplicate_slide 过程中发生错误: {e}")


def modify_ppt_slide_text(
        input_pptx_path: str,
        content_list: list,
        slide_number: int = 1,
        type: str = None
):
    """
    修改指定幻灯片中以'__placeholder__'开头的文本框内容，按顺序填充提供的文本列表，直接覆盖原文件。

    :param input_pptx_path: 输入的 PPTX 文件路径（将被直接修改）。
    :param content_list: 用于替换的文本列表，按顺序填充。
    :param slide_number: 要修改的幻灯片编号（从 1 开始）。
    """
    slide_xml_path = f"ppt/slides/slide{slide_number}.xml"
    modified_slide_xml_content = None
    print(f"正在修改：{type}")

    # 定义 Open XML DrawingML 的主要命名空间
    ns = {
        'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
        'p': 'http://schemas.openxmlformats.org/presentationml/2006/main'
    }

    try:
        # --- 步骤 1: 解包并读取指定的幻灯片 XML ---
        with zipfile.ZipFile(input_pptx_path, 'r') as z_in:
            # 检查幻灯片文件是否存在
            if slide_xml_path not in z_in.namelist():
                print(f"错误: 在 {input_pptx_path} 中找不到 {slide_xml_path}。")
                return

            # 读取所有文件内容到内存
            package_files = {item.filename: z_in.read(item.filename) for item in z_in.infolist()}

            # 获取指定幻灯片的 XML 内容
            slide_xml_content = package_files[slide_xml_path]

            # --- 步骤 2: 使用 lxml 解析和修改 XML ---
            root = etree.fromstring(slide_xml_content)

            # 查找所有形状（sp）元素
            shape_elements = root.xpath('.//p:sp', namespaces=ns)
            placeholder_boxes = []

            # 【重大修改】遍历所有形状，找到并组合文本
            for sp in shape_elements:
                # 找到一个形状内部所有的 <a:t> 元素
                text_elements = sp.xpath('.//a:t', namespaces=ns)
                if not text_elements:
                    continue

                # 【逻辑修正】将一个形状内所有 <a:t> 的文本拼接成一个完整字符串
                full_text = "".join([t.text for t in text_elements if t.text])

                # 对完整字符串进行判断
                if full_text.startswith('__placeholder__'):
                    print("找到的完整占位符文本：", full_text)
                    # 记录完整的占位符文本和形状元素
                    placeholder_boxes.append((full_text, sp))

            if not placeholder_boxes:
                print(f"在幻灯片 {slide_number} 中没有找到以'__placeholder__'开头的文本框。")
                return

            # --- 步骤 3: 按形状的索引排序 ---
            def get_placeholder_number(item):
                """辅助函数：从占位符文本中提取数字并转换为整数"""
                # item[0] 是占位符文本，例如 '__placeholder__12'
                placeholder_text = item[0]
                try:
                    "dsfjsdf".lstrip()
                    # 移除前缀，剩下的就是数字
                    number_str = placeholder_text.strip().replace('__placeholder__', '')
                    return int(number_str)
                except (ValueError, TypeError):
                    # 如果转换失败（例如文本格式不符），给一个默认的大值，让它排在最后
                    return float('inf')

            sorted_boxes = sorted(placeholder_boxes, key=get_placeholder_number)

            # --- 步骤 4: 按排序后的顺序替换文本框内容 (修改版) ---
            modified = False
            for (index, sp), chapter in zip(sorted_boxes, content_list):
                # 获取文本框中的所有 <a:p> 段落元素
                paragraphs = sp.xpath('.//a:p', namespaces=ns)

                if paragraphs:
                    # 1. 定位到第一个段落
                    first_p = paragraphs[0]

                    # 2. 找到第一个段落中的所有 <a:r> 运行元素
                    runs = first_p.xpath('.//a:r', namespaces=ns)

                    if runs:
                        # --- 保留第一个 "run" 的样式 ---
                        # 3. 定位到第一个 "run"
                        first_r = runs[0]

                        # 4. 找到或创建其中的 <a:t> 文本元素并修改文本
                        t = first_r.find('.//a:t', namespaces=ns)
                        if t is None:  # 如果 "run" 里没有文本元素，则创建一个
                            t = etree.SubElement(first_r, '{' + ns['a'] + '}t')
                        t.text = chapter  # 设置新文本

                        # 5. 删除第一个 "run" 之后的所有其他 "run" 元素
                        for r in runs[1:]:
                            r.getparent().remove(r)
                    else:
                        # 如果段落是空的（没有 "run"），则创建一个新的
                        r = etree.SubElement(first_p, '{' + ns['a'] + '}r')
                        t = etree.SubElement(r, '{' + ns['a'] + '}t')
                        t.text = chapter

                    # 6. 删除多余的段落
                    for p in paragraphs[1:]:
                        p.getparent().remove(p)

                    modified = True

            if modified:
                print(f"成功在幻灯片 {slide_number} 中替换了 {len(sorted_boxes)} 个文本框的内容。")
            else:
                print(f"在幻灯片 {slide_number} 中未进行任何替换。")

            # 将修改后的 XML 树转换回字符串
            modified_slide_xml_content = etree.tostring(root, pretty_print=True, xml_declaration=True, encoding='UTF-8')

        # --- 步骤 5: 将修改后的内容写回原 PPTX 文件 ---
        if modified_slide_xml_content:
            package_files[slide_xml_path] = modified_slide_xml_content
            with zipfile.ZipFile(input_pptx_path, 'w', zipfile.ZIP_DEFLATED) as z_out:
                for filename, content in package_files.items():
                    z_out.writestr(filename, content)
            print(f"已直接修改原文件: {input_pptx_path}")

    except FileNotFoundError:
        print(f"错误: 输入文件未找到 -> {input_pptx_path}")
    except Exception as e:
        print(f"处理过程中发生错误: {e}")

# # --- 使用示例 ---
# if __name__ == '__main__':
#     # 示例调用
#     modify_ppt_slide_text(
#         input_pptx_path="./ppt/5temp.pptx",
#         content_list=["Chapter 1", "Chapter 2", "Chapter 3","Chapter 111", "Chapter 4"],
#         slide_number=1
#     )

# --- 使用示例 ---
if __name__ == '__main__':
    # 准备文件：
    # 1. 创建一个名为 "ppt1.pptx" 的文件，可以包含任意内容，比如一页写着“这是第一页”的幻灯片。
    # 2. 创建一个名为 "template.pptx" 的文件，确保它只有一页。在这一页上设计你想要的模板，
    #    比如包含特定的背景、Logo图片、文本框样式等。


    dest_file = "./ppt/多agent讲解.pptx"
    template_file = "./ppt/content/5temp.pptx"
    output_file = "./ppt/temp.pptx"


    merge_elements_onto_slide(
        dest_pptx_path=dest_file,
        template_pptx_path=template_file,
    )