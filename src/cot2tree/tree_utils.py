import json
import numpy as np
import networkx as nx
import plotly.graph_objects as go



class TreeNode:
    def __init__(self, value, level, text=None, is_critical=False, father=None, children=None, cate=None, thought_list=None):
        self.value = value
        self.level = level
        self.text = text
        self.is_critical = is_critical
        self.cate = cate
        self.father = father
        self.thought_list = thought_list
        self.children = children if children is not None else []


def tree_to_dict(node):
    return {
        "value": node.value,
        "level": node.level,
        "children": [tree_to_dict(child) for child in node.children]
    }

def dict_to_tree_with_cate(tree_dict, father=None):
    value = tree_dict["value"]
    level = tree_dict["level"]
    cate = tree_dict["cate"]
    thought_list = tree_dict["thought_list"]
    node = TreeNode(value, level, father=father, cate=cate, thought_list=thought_list)
    node.children = [dict_to_tree_with_cate(child_dict, node) for child_dict in tree_dict["children"]]
    return node

def dict_to_tree(tree_dict, father=None):
    value = tree_dict["value"]
    level = tree_dict["level"]
    node = TreeNode(value, level, father=father)
    node.children = [dict_to_tree(child_dict, node) for child_dict in tree_dict["children"]]
    return node


def tree_to_dict_with_cate(node):
    return {
        "value": node.value,
        "level": node.level,
        "cate": node.cate,
        "thought_list": node.thought_list,
        "children": [tree_to_dict_with_cate(child) for child in node.children]
    }


def dict_to_tree_with_text(tree_dict, text_list, father=None):
    value = tree_dict["value"]
    level = tree_dict["level"]
    cate = tree_dict["cate"]
    thought_list = tree_dict["thought_list"]
    node_v = value.split("-")[0] if value!=0 else "0"
    text = text_list[node_v]
    node = TreeNode(value, level, text=text, cate=cate, thought_list=thought_list, father=father)
    node.children = [dict_to_tree_with_text(child_dict, text_list, node) for child_dict in tree_dict["children"]]
    return node

def save_tree_to_file(node, file_path):
    try:
        tree_dict = tree_to_dict(node)
        with open(file_path, 'w') as file:
            json.dump(tree_dict, file, indent=4)
        print(f"树结构数据已成功保存到 {file_path}")
    except Exception as e:
        print(f"保存文件时出错: {e}")


def convert_tree_to_string(node):
    tree_dict = tree_to_dict(node)
    tree_string = json.dumps(tree_dict)
    return tree_string


def load_tree_from_file(file_path):
    try:
        with open(file_path, 'r') as file:
            tree_dict = json.load(file)
        return dict_to_tree(tree_dict)
    except FileNotFoundError:
        print("错误: 文件未找到!")
    except Exception as e:
        print(f"错误: 发生了一个未知错误: {e}")
    return None

def tree_to_coordinates(root):
    nodes = {}
    edges = []
    
    def dfs(node):
        max_level = node.level
        for child in node.children:
            max_level = max(dfs(child), max_level)
        # print(node.value, max_level)
        return max_level

    max_level = dfs(root)
    level_curr_pos = {i:0 for i in range(max_level+1)}
    
    def place_nodes(node):
        
        node_json = json.dumps(tree_to_dict(node))

        if len(node.children) == 0:
            curr = level_curr_pos[node.level]
            
            bro_max_x = -1000
            for bro in node.father.children:
                bro_json = json.dumps(tree_to_dict(bro))
                if bro_json in nodes:
                    bro_max_x = max(bro_max_x, nodes[bro_json][0])
            # print(node.value, bro_max_x)
            x = max(curr, bro_max_x+1)
            y = -node.level*1.1-1
            nodes[node_json] = (x, y, node.value, node.text, node.is_critical, node.level)

            level_curr_pos[node.level] = x+1

            for key in level_curr_pos:
                level_curr_pos[key] = x+1
            return

        x = []
        for child in node.children:
            place_nodes(child)
            child_json = json.dumps(tree_to_dict(child))
            x.append(nodes[child_json][0])
            
        max_pos = max(x)+1
        for key in level_curr_pos:
            level_curr_pos[key] = max(level_curr_pos[key], max_pos)
        x = np.mean(x)
        y = -node.level*1.1-1
        nodes[node_json] = (x, y, node.value, node.text, node.is_critical, node.level)
        
        for child in node.children:
            child_json = json.dumps(tree_to_dict(child))
            # print(child.cate)
            edges.append((nodes[node_json][0], nodes[node_json][1], nodes[child_json][0], nodes[child_json][1], child.cate))
        return 

    place_nodes(root)
    # print(nodes.values())
    return list(nodes.values()), edges



def is_overlapping(edge1, edges):
    def orientation(p, q, r):
        """
        计算向量 pq 和向量 pr 的叉积，判断点 r 相对于向量 pq 的位置
        :param p: 第一个点
        :param q: 第二个点
        :param r: 第三个点
        :return: 0 表示共线，1 表示顺时针，2 表示逆时针
        """
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0:
            return 0
        return 1 if val > 0 else 2

    def on_segment(p, q, r):
        """
        检查点 q 是否在线段 pr 上
        :param p: 线段起点
        :param q: 待检查点
        :param r: 线段终点
        :return: 如果点 q 在线段 pr 上返回 True，否则返回 False
        """
        return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
                q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))

    x1, y1, x2, y2, _ = edge1
    for edge in edges:
        x3, y3, x4, y4, _ = edge
        if (x1, y1) == (x3, y3) or (x1, y1) == (x4, y4) or (x2, y2) == (x3, y3) or (x2, y2) == (x4, y4):
            continue
        # 计算四个方向的叉积
        o1 = orientation((x1, y1), (x2, y2), (x3, y3))
        o2 = orientation((x1, y1), (x2, y2), (x4, y4))
        o3 = orientation((x3, y3), (x4, y4), (x1, y1))
        o4 = orientation((x3, y3), (x4, y4), (x2, y2))

        # 一般情况：两条线段相交
        if o1 != o2 and o3 != o4:
            return True

        # 特殊情况：共线且重叠
        if o1 == 0 and on_segment((x1, y1), (x3, y3), (x2, y2)):
            return True
        if o2 == 0 and on_segment((x1, y1), (x4, y4), (x2, y2)):
            return True
        if o3 == 0 and on_segment((x3, y3), (x1, y1), (x4, y4)):
            return True
        if o4 == 0 and on_segment((x3, y3), (x2, y2), (x4, y4)):
            return True

    return False


# def tree_to_coordinates(root):
#     nodes = []
#     edges = []
#     node_positions = {}
#     level_width = {}  # 记录每一层的节点数量
#     max_width = [0]

#     def calculate_level_width(node):
#         if node.level not in level_width:
#             level_width[node.level] = 0
#         level_width[node.level] += 1
#         max_width[0] = max(max_width[0], level_width[node.level]) 
#         for child in node.children:
#             calculate_level_width(child)
#     calculate_level_width(root)
#     # print(level_width)
    
#     total_width = [max_width[0]+1*i for i in range(len(level_width)+1)]
#     level_curr_width = {}  # 记录每一层的节点数量
#     def place_nodes(node, x, y):
#         nodes.append((x, y, node.value, node.text, node.is_critical, node.level))
#         node_positions[node] = (x, y)
#         child_index = 0
#         max_x = -100000
#         for child in node.children:
#             # 计算子节点的 x 坐标，根据该层的节点数量均匀分布
#             curr_level = child.level
#             if curr_level not in level_curr_width:
#                 level_curr_width[curr_level] = -int(total_width[curr_level]/2)  

#             child_y = - curr_level-1
#             if level_curr_width[curr_level] > max_x:
#                 child_x = level_curr_width[curr_level]
#                 max_x = child_x
#             else:
#                 child_x = max_x + 1

#             # cnt = 1
#             # while is_overlapping((x, y, child_x, child_y), edges) and cnt < 10:
#             #     child_x += 1
#             #     cnt += 1
                
#             max_x = child_x
#             level_width[curr_level] -= 1                
#             if level_width[curr_level] == 0:
#                 level_curr_width[curr_level] = total_width[curr_level]
#             else:
#                 if total_width[curr_level]-child_x < level_width[curr_level]:
#                     # print(total_width[curr_level:])
#                     # print( (child_x + level_width[curr_level] - total_width[curr_level]))
#                     total_width[curr_level:] = [i + int(child_x + level_width[curr_level] - total_width[curr_level]) for i in total_width[curr_level:]]
#                 level_curr_width[curr_level] += (total_width[curr_level]-child_x)/level_width[curr_level]
#             # print(child.value, child_x, level_curr_width[curr_level], max_x)
#             edges.append((x, y, child_x, child_y))
#             place_nodes(child, child_x, child_y)
#             child_index += 1
#     place_nodes(root, 0, -1)
#     # print(level_curr_width)

#     return nodes, edges




# 生成可视化的 HTML 文件
def visualize_tree(root, output_file, level_texts=None, item=None, edge_color_info=None):
    nodes, edges = tree_to_coordinates(root)

    # 提取节点的坐标、值、文本和是否关键信息
    node_x = [x for x, y, _, _, _, _ in nodes]
    node_y = [y for x, y, _, _, _, _ in nodes]
    edge_color_info = edge_color_info if edge_color_info is not None else [0.5 for i in range(len(edges))]
    node_values = [value for _, _, value, _, _,_ in nodes]
    node_texts = ['<br>'.join([text[i:i+50] for i in range(0, len(text), 50)]) for _, _, _, text, _, _ in nodes]
    node_colors = ["blue" if is_critical else "#686789" for _, _, _, _, is_critical, _ in nodes]
    node_levels = [f"Step {level}" for level in range(len(level_texts)+1)]
    node_levels_x = [max(node_x)+1.5 for level in range(len(level_texts)+1)]
    node_levels_y = [-1-level*1.1 for level in range(len(level_texts)+1)]
    if level_texts is not None:
        level_info = 'text'
        node_levels_text = []
        for level in range(len(level_texts)+1):
            if level == 0:
                node_levels_text.append("")
                continue
            text = level_texts[level]
            node_levels_text.append('<br>'.join([text[i:i+50] for i in range(0, len(text), 50)]))
    # print(node_levels_text)

    # 提取边的坐标
    edge_x = []
    edge_y = []
    edge_c = []
    for i, (x0, y0, x1, y1, c) in enumerate(edges):
        edge_x.extend([x0, x1])
        edge_y.extend([y0, y1])
        edge_c.append(c)

    fig = go.Figure()
    offset = 0.03  # Adjust this value to control the offset distance
    # Add parent to child edges (solid lines)
    for i in range(len(edges)):
        start_idx = i * 2
        end_idx = start_idx + 2
        # print(edge_c[i])
        if edge_c[i] == 2:
            rgba_color = f'hsl(10, 50%, {80-0.35*int(edge_color_info[start_idx]*100)}%)'
        if edge_c[i] == 3:
            rgba_color = f'hsl(240, 40%, {80-0.35*int(edge_color_info[start_idx]*100)}%)'
        if edge_c[i] == 4:
            rgba_color = f'hsl(40, 40%, {80-0.35*int(edge_color_info[start_idx]*100)}%)'
        else:
            rgba_color = f'hsl(0, 0%, {80-0.35*int(edge_color_info[start_idx]*100)}%)'
        # rgba_color = f'hsl(0, 0%, {100-int(edge_color_info[start_idx]*100)}%)'
        # Add slight offset to parent-to-child edges
        x_coords = edge_x[start_idx:end_idx]
        y_coords = edge_y[start_idx:end_idx]
        # Calculate the offset based on the angle of the edge
        dx = x_coords[1] - x_coords[0]
        dy = y_coords[1] - y_coords[0]
        length = (dx**2 + dy**2)**0.5
        if length > 0:  # Avoid division by zero
            x_coords = [x + offset * dy/length for x in x_coords]
            y_coords = [y - offset * dx/length for y in y_coords]
        
        edge_trace = go.Scatter(
            x=x_coords, 
            y=y_coords,
            line=dict(color=rgba_color, width=0.6+2*edge_color_info[start_idx]),
            hoverinfo='none',
            mode='lines'
        )
        fig.add_trace(edge_trace)
    
        if edge_c[i] == 2:
            rgba_color = f'hsl(10, 50%, {80-0.35*int(edge_color_info[start_idx+1]*100)}%)'
        elif edge_c[i] == 3:
            rgba_color = f'hsl(240, 40%, {80-0.35*int(edge_color_info[start_idx+1]*100)}%)'
        elif edge_c[i] == 4:
            rgba_color = f'hsl(40, 40%, {80-0.35*int(edge_color_info[start_idx+1]*100)}%)'
        else:
            rgba_color = f'hsl(0, 0%, {80-0.35*int(edge_color_info[start_idx+1]*100)}%)'
        # rgba_color = f'hsl(0, 0%, {100-int(edge_color_info[start_idx+1]*100)}%)'
        if length > 0:  # Avoid division by zero
            
            x_coords = [x - offset * dy/length for x in x_coords]
            y_coords = [y + offset * dx/length for y in y_coords]
        # print(rgba_color)
        child_to_parent_trace = go.Scatter(
            x=x_coords, 
            y=y_coords,
            line=dict(color=rgba_color, width=0.6+2*edge_color_info[start_idx+1], dash='dash'),
            # line=dict(color='black', width=0.6, dash='dash'),
            hoverinfo='none',
            mode='lines'
        )
        fig.add_trace(child_to_parent_trace)

    # 创建节点的绘图对象
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_values,
        hovertext=node_texts,
        hoverinfo='text',
        marker=dict(
            size=40,  # 增大节点大小
            color=node_colors,
            line=dict(width=1.5, color='black')
        ),
        textposition="middle center",
        textfont=dict(color='white', weight='bold')  
    )
    fig.add_trace(node_trace)
    level_trace = go.Scatter(
        x=node_levels_x,  # 将 level 信息显示在树的右侧
        y=node_levels_y,
        mode='text',
        text=node_levels,
        textposition="middle left",
        textfont=dict(color='black',size=14,weight='bold'),
        hoverinfo=level_info,
        hovertext=node_levels_text,
    )
    fig.add_trace(level_trace)

    # if "full_prompt" in item:
    #     query = '<br>'.join([item["full_prompt"][i:i+50] for i in range(0, len(item["full_prompt"]), 50)])
    # else:
    query= 'none'
    question = go.Scatter(
        x=[min(node_x)+1],  # 将 level 信息显示在树的右侧
        y=[max(node_y)],
        mode='markers+text',
        text=["Query"],
        hovertext=[query],
        hoverinfo='text',
        marker=dict(
            size=40,
            color=["#849b91"],
            line=dict(width=1.5, color='black')
        ),
        textposition="middle center",
        textfont=dict(color='white', weight='bold')    
    )
    fig.add_trace(question)

    score = go.Scatter(
        x=[min(node_x)-1],  # 将 level 信息显示在树的右侧
        y=[max(node_y)],
        mode='markers+text',
        text=["Score"],
        hovertext=[item["score"]],
        hoverinfo='text',
        marker=dict(
            size=40,
            color=["#9aa690" if item["score"] == "1" else "#beb1a8"],
            line=dict(width=1.5, color='black')
        ),
        textposition="middle center",
        textfont=dict(color='white', weight='bold')    
    )
    fig.add_trace(score)

    answer = go.Scatter(
        x=[min(node_x)-1],  # 将 level 信息显示在树的右侧
        y=[max(node_y)-1],
        mode='markers+text',
        text=["Ans"],
        hovertext=['<br>'.join([item["gold"][0][i:i+50] for i in range(0, len(item["gold"][0]), 50)])],
        hoverinfo='text',
        marker=dict(
            size=40,
            color=["#849b91"],
            line=dict(width=1.5, color='black')
        ),
        textposition="middle center",
        textfont=dict(color='white', weight='bold')   
    )
    fig.add_trace(answer)

    # 设置图形布局
    fig.update_layout(
        showlegend=False,
        hovermode='closest',
        plot_bgcolor='#f3eeea',
        # paper_bgcolor='white',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )

    # 保存为 HTML 文件
    fig.write_html(output_file)