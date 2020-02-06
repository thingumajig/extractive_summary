from nltk import Tree


def tok_format(t):
  return f'{t.orth_}-{t.dep_}'


def to_nltk_tree(node):
  if node.n_lefts + node.n_rights > 0:
    return Tree(tok_format(node), [to_nltk_tree(child) for child in node.children])
  else:
    return tok_format(node)


def print_tree(node):
  to_nltk_tree(node).pretty_print()
