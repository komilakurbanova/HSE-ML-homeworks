import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):
    """
    РџРѕРґ РєСЂРёС‚РµСЂРёРµРј Р”Р¶РёРЅРё Р·РґРµСЃСЊ РїРѕРґСЂР°Р·СѓРјРµРІР°РµС‚СЃСЏ СЃР»РµРґСѓСЋС‰Р°СЏ С„СѓРЅРєС†РёСЏ:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ вЂ” РјРЅРѕР¶РµСЃС‚РІРѕ РѕР±СЉРµРєС‚РѕРІ, $R_l$ Рё $R_r$ вЂ” РѕР±СЉРµРєС‚С‹, РїРѕРїР°РІС€РёРµ РІ Р»РµРІРѕРµ Рё РїСЂР°РІРѕРµ РїРѕРґРґРµСЂРµРІРѕ,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ вЂ” РґРѕР»СЏ РѕР±СЉРµРєС‚РѕРІ РєР»Р°СЃСЃР° 1 Рё 0 СЃРѕРѕС‚РІРµС‚СЃС‚РІРµРЅРЅРѕ.

    РЈРєР°Р·Р°РЅРёСЏ:
    * РџРѕСЂРѕРіРё, РїСЂРёРІРѕРґСЏС‰РёРµ Рє РїРѕРїР°РґР°РЅРёСЋ РІ РѕРґРЅРѕ РёР· РїРѕРґРґРµСЂРµРІСЊРµРІ РїСѓСЃС‚РѕРіРѕ РјРЅРѕР¶РµСЃС‚РІР° РѕР±СЉРµРєС‚РѕРІ, РЅРµ СЂР°СЃСЃРјР°С‚СЂРёРІР°СЋС‚СЃСЏ.
    * Р’ РєР°С‡РµСЃС‚РІРµ РїРѕСЂРѕРіРѕРІ, РЅСѓР¶РЅРѕ Р±СЂР°С‚СЊ СЃСЂРµРґРЅРµРµ РґРІСѓС… СЃРѕСЃРґРµРЅРёС… (РїСЂРё СЃРѕСЂС‚РёСЂРѕРІРєРµ) Р·РЅР°С‡РµРЅРёР№ РїСЂРёР·РЅР°РєР°
    * РџРѕРІРµРґРµРЅРёРµ С„СѓРЅРєС†РёРё РІ СЃР»СѓС‡Р°Рµ РєРѕРЅСЃС‚Р°РЅС‚РЅРѕРіРѕ РїСЂРёР·РЅР°РєР° РјРѕР¶РµС‚ Р±С‹С‚СЊ Р»СЋР±С‹Рј.
    * РџСЂРё РѕРґРёРЅР°РєРѕРІС‹С… РїСЂРёСЂРѕСЃС‚Р°С… Р”Р¶РёРЅРё РЅСѓР¶РЅРѕ РІС‹Р±РёСЂР°С‚СЊ РјРёРЅРёРјР°Р»СЊРЅС‹Р№ СЃРїР»РёС‚.
    * Р—Р° РЅР°Р»РёС‡РёРµ РІ С„СѓРЅРєС†РёРё С†РёРєР»РѕРІ Р±Р°Р»Р» Р±СѓРґРµС‚ СЃРЅРёР¶РµРЅ. Р’РµРєС‚РѕСЂРёР·СѓР№С‚Рµ! :)

    :param feature_vector: РІРµС‰РµСЃС‚РІРµРЅРЅРѕР·РЅР°С‡РЅС‹Р№ РІРµРєС‚РѕСЂ Р·РЅР°С‡РµРЅРёР№ РїСЂРёР·РЅР°РєР°
    :param target_vector: РІРµРєС‚РѕСЂ РєР»Р°СЃСЃРѕРІ РѕР±СЉРµРєС‚РѕРІ,  len(feature_vector) == len(target_vector)

    :return thresholds: РѕС‚СЃРѕСЂС‚РёСЂРѕРІР°РЅРЅС‹Р№ РїРѕ РІРѕР·СЂР°СЃС‚Р°РЅРёСЋ РІРµРєС‚РѕСЂ СЃРѕ РІСЃРµРјРё РІРѕР·РјРѕР¶РЅС‹РјРё РїРѕСЂРѕРіР°РјРё, РїРѕ РєРѕС‚РѕСЂС‹Рј РѕР±СЉРµРєС‚С‹ РјРѕР¶РЅРѕ
     СЂР°Р·РґРµР»РёС‚СЊ РЅР° РґРІРµ СЂР°Р·Р»РёС‡РЅС‹Рµ РїРѕРґРІС‹Р±РѕСЂРєРё, РёР»Рё РїРѕРґРґРµСЂРµРІР°
    :return ginis: РІРµРєС‚РѕСЂ СЃРѕ Р·РЅР°С‡РµРЅРёСЏРјРё РєСЂРёС‚РµСЂРёСЏ Р”Р¶РёРЅРё РґР»СЏ РєР°Р¶РґРѕРіРѕ РёР· РїРѕСЂРѕРіРѕРІ РІ thresholds len(ginis) == len(thresholds)
    :return threshold_best: РѕРїС‚РёРјР°Р»СЊРЅС‹Р№ РїРѕСЂРѕРі (С‡РёСЃР»Рѕ)
    :return gini_best: РѕРїС‚РёРјР°Р»СЊРЅРѕРµ Р·РЅР°С‡РµРЅРёРµ РєСЂРёС‚РµСЂРёСЏ Р”Р¶РёРЅРё (С‡РёСЃР»Рѕ)
    """
    data = np.column_stack((feature_vector, target_vector))
    sorted_data = data[data[:, 0].argsort()]
    target_labels = sorted_data[:, 1]

    unique_features, unique_counts = np.unique(feature_vector, return_counts=True)
    thresholds = ((unique_features + np.concatenate([unique_features[1:], [0]])) / 2)[:-1]

    cumsum_labels = np.cumsum(target_labels)
    total_size = len(target_vector)
    
    left_sizes = np.arange(1, total_size)
    right_sizes = total_size - left_sizes
    
    left_cumsum = cumsum_labels[:-1]
    right_cumsum = cumsum_labels[-1] - left_cumsum

    left_prob_0 = left_cumsum / left_sizes
    left_prob_1 = 1 - left_prob_0
    right_prob_0 = (right_cumsum / right_sizes)
    right_prob_1 = 1 - right_prob_0

    impurity_left = 1 - left_prob_0**2 - left_prob_1**2
    impurity_right = 1 - right_prob_0**2 - right_prob_1**2
    
    ginis = - (left_sizes / total_size) * impurity_left - (right_sizes / total_size) * impurity_right
    ginis = ginis[np.cumsum(unique_counts)[:-1] - 1]
    
    best_idx = np.argmax(ginis)
    gini_best = ginis[best_idx]
    threshold_best = thresholds[best_idx]
    
    return thresholds, ginis, threshold_best, gini_best


class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, depth):
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return
        
        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_click / current_count
                sorted_categories = list(map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1])))
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))

                feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature])))
            else:
                raise ValueError
            
            if len(np.unique(feature_vector)) < 2 or np.all(feature_vector == feature_vector[0]):
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            if gini_best is None or gini > gini_best:
                split = feature_vector < threshold
                if self._min_samples_leaf is None or (self._min_samples_leaf is not None and 
                                                      np.sum(split) >= self._min_samples_leaf and 
                                                      np.sum(~split) >= self._min_samples_leaf):
                    feature_best = feature
                    gini_best = gini
                    if feature_type == "real":
                        threshold_best = threshold
                    elif feature_type == "categorical":
                        threshold_best = list(map(lambda x: x[0], filter(lambda x: x[1] < threshold, categories_map.items())))
                    else:
                        raise ValueError

        if feature_best is None or depth == self._max_depth or (self._min_samples_split is not None and self._min_samples_split > len(sub_y)):
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth + 1)
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"], depth + 1)

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]

        feature = node["feature_split"]

        if self._feature_types[feature] == "real":
            if x[feature] < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            return self._predict_node(x, node["right_child"])

        elif self._feature_types[feature] == "categorical":
            if x[feature] in node["categories_split"]:
                return self._predict_node(x, node["left_child"])
            return self._predict_node(x, node["right_child"])
        else:
            raise ValueError

    def fit(self, X, y):
        self._fit_node(X, y, self._tree, 0)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
    
    def get_params(self, deep=False):
        return {
            "feature_types": self._feature_types,
            "max_depth": self._max_depth,
            "min_samples_split": self._min_samples_split,
            "min_samples_leaf": self._min_samples_leaf,
        }
    
    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self

    def get_tree_size(self):
        return len(self._tree)
