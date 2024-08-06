# 实现 关系 自动对齐，两个实体对齐一个关系：利用对齐的实体，尝试对齐关系
def load_triples(file_name, entity, matrix, map):
    triples = []
    for line in open(file_name, 'r'):
        para = line.split()
        if len(para) == 5:
            head, r, tail, ts, te = [str(item) for item in para]
            head = int(head)
            tail = int(tail)
            r = int(r)

            if (head in entity and tail in entity):
                triples.append((head, r, tail, ts, te))
                matrix[map[head]][map[tail]].append(r)

    return triples


def load_alignment_pair(file_name):
    alignment_pair = []
    for line in open(file_name, 'r'):
        e1, e2 = line.split()
        alignment_pair.append((int(e1), int(e2)))
    return alignment_pair


def load_data(lang, train_ratio=0.3):
    map = {}
    train_pair = load_alignment_pair(lang + 'sup_pairs')
    dev_pair = load_alignment_pair(lang + 'ref_pairs')


    if train_ratio < 0.3:
        lens = len(train_pair) + len(dev_pair)
        train_ratio = 2000
        train_pair = train_pair[:train_ratio]
        print(len(train_pair))
    entity1 = set()
    entity2 = set()
    length = len(train_pair)

    for i in range(length):
        entity1.add(int(train_pair[i][0]))
        entity2.add(int(train_pair[i][1]))
        map[int(train_pair[i][0])] = i
        map[int(train_pair[i][1])] = i

    matrix1 = [[[] for _ in range(length)] for _ in range(length)]
    matrix2 = [[[] for _ in range(length)] for _ in range(length)]

    # 加载三元组所有信息
    triples1 = load_triples(lang + 'triples_1', entity1, matrix1, map)
    triples2 = load_triples(lang + 'triples_2', entity2, matrix2, map)

    return matrix1, matrix2, length

# 找到所有相关的三元组信息
matrix1, matrix2, length = load_data('data/ICEWS05-15/', 0.1)  # YAGO-WIKI50K


matches = []
for i in range(length):
    for j in range(length):
        if len(matrix1[i][j]) > 0 and len(matrix2[i][j]) > 0:
            matches.append((matrix1[i][j], matrix2[i][j]))
# print(matches)


# 初始化一个空字典用于计数
count_dict = {}
# 遍历 matches 列表，对每个元素进行计数
for match in matches:
    combinations = [(x, y) for x in match[0] for y in match[1]]
    for match_key in combinations:
        if match_key in count_dict:
            count_dict[match_key] += 1  # 如果元素已经在字典中，计数加1
        else:
            count_dict[match_key] = 1  # 如果元素不在字典中，初始化计数为1


# 将字典items转换为列表，并根据count次数进行排序
sorted_count_dict = sorted(count_dict.items(), key=lambda item: item[1], reverse=True)
# 打印每个元素及其出现次数
for key, count in sorted_count_dict:
    if (count > 1):
        print(f"{key} 出现 {count} 次")

# 计算每个独立关系出现的总次数
individual_relation_count = {}
for (relation1, relation2), count in count_dict.items():
    if relation1 not in individual_relation_count:
        individual_relation_count[relation1] = count
    else:
        individual_relation_count[relation1] += count
    if relation2 not in individual_relation_count:
        individual_relation_count[relation2] = count
    else:
        individual_relation_count[relation2] += count


# 设置两个集合来跟踪已经处理过的关系
processed_relations1 = set()
processed_relations2 = set()
# 初始化编号从1开始
relation_id = 1
# 初始化一个字典来跟踪已经分配过的关系编号
assigned_relations = {}
# 打开文件用于写入结果
with open('rels_sim', 'w') as f:
    for key, count in sorted_count_dict:
        relation1, relation2 = key
        # 检查是否满足第一个标准：出现次数大于100
        if count > 100:
            # 检查是否满足第二个标准：出现次数大于重复关系总次数的30%
            if (count > 0.3 * individual_relation_count[relation1]) and (count > 0.3 * individual_relation_count[relation2]):
                # 检查关系是否已经有编号
                if relation1 in assigned_relations:
                    assigned_id = assigned_relations[relation1]
                    # 将重复关系编号写入文件

                    f.write(str(relation2) + "\t" + str(assigned_id))
                    f.write("\n")
                elif relation2 in assigned_relations:
                    assigned_id = assigned_relations[relation2]
                    # 将重复关系编号写入文件
                    f.write(str(relation1) + "\t" + str(assigned_id))
                    f.write("\n")

                else:
                    # 将新的对齐关系和编号写入文件
                    f.write(str(relation1) + "\t" + str(relation_id))
                    f.write("\n")
                    f.write(str(relation2) + "\t" + str(relation_id))
                    f.write("\n")
                    # 在字典中记录下分配的编号
                    assigned_relations[relation1] = relation_id
                    assigned_relations[relation2] = relation_id
                    # 更新编号
                    relation_id += 1