

class Template:
    def __init__(self, in_t: str, out_t: str, input_fields, output_fields, template_id):
        self.input_template = in_t
        self.output_template = out_t
        self.template_id = template_id
        self.task = template_id.split('-')[0]

        for _ in input_fields:
            for __ in _.split('/'):
                assert __ in ['history', 'item', 'preference', 'vague_intention', 'specific_intention', 'candidate_items', 'category']
        for _ in output_fields:
            for __ in _.split('/'):
                assert __ in ['item', 'history', 'preference', 'specific_intention', 'vague_intention']

        self.input_fields = input_fields
        self.output_fields = output_fields

    def get_input_text(self, input_args: dict):
        return self.input_template.format_map(input_args)

    def get_output_text(self, output_args: dict):
        return self.output_template.format_map(output_args)


TradRec_task_key = 'TradRec'
ProductSearch_task_key = 'ProductSearch'
PersonalizedSearch_task_key = 'PersonalizedSearch'

ValTradRec_task_key = 'ValTradRec'
ValFullRec_task_key = 'ValFullRec'
ValProductSearch_task_key = 'ValProductSearch'
ValPersonalizedSearch_task_key = 'ValPersonalizedSearch'

TestTradRec_task_key = 'TestTradRec'
TestFullRec_task_key = 'TestFullRec'
TestProductSearch_task_key = 'TestProductSearch'
TestPersonalizedSearch_task_key = 'TestPersonalizedSearch'
################################################################################################################
#                                                                                                              #
#                               traditional recommendation templates                                           #
#                                                                                                              #
################################################################################################################


TradRec_group = {}

template = {
    'in_t': "Using the user’s historical interactions as input data, " +
            "predict the next product that the user is most likely to interact with. " +
            "The historical interactions are provided as follows: {history}.",
    'out_t': "{item}",
    'input_fields': ['history'],
    'output_fields': ['item'],
    'template_id': f"{TradRec_task_key}-{len(TradRec_group)}",
}
TradRec_group[f"{TradRec_task_key}-{len(TradRec_group)}"] = Template(**template)


template = {
    'in_t': "Recommend the next potential product to a user based on his profile and past interaction. " +
            "You have access to the user’s profile information, " +
            "including his preference: {preference} and past interactions: {history}. " +
            "Now you need to determine what product would be recommended to him.",
    'out_t': "{item}",
    'input_fields': ['preference', 'history'],
    'output_fields': ['item'],
    'template_id': f"{TradRec_task_key}-{len(TradRec_group)}",
}
TradRec_group[f"{TradRec_task_key}-{len(TradRec_group)}"] = Template(**template)


template = {
    'in_t': "Like reasoning. Here is some information about the user, such as his historical interactions: {history}. " +
            "Based on this information, your task is to infer the user’s preference based on his historical interactions " +
            "and recommend the next product to the user.",
    'out_t': "{preference} {item}",
    'input_fields': ['history'],
    'output_fields': ['preference', 'item'],
    'template_id': f"{TradRec_task_key}-{len(TradRec_group)}",
}
TradRec_group[f"{TradRec_task_key}-{len(TradRec_group)}"] = Template(**template)


template = {
    'in_t': "Given the following historical interaction of the user: {history}. " +
            "You can infer the user’s preference. {preference}. " +
            "Please predict next possible item for the user.",
    'out_t': "{item}",
    'input_fields': ['history', 'preference'],
    'output_fields': ['item'],
    'template_id': f"{TradRec_task_key}-{len(TradRec_group)}",
}
TradRec_group[f"{TradRec_task_key}-{len(TradRec_group)}"] = Template(**template)


template = {
    'in_t': "Based on the historical interactions shown below: {history}, "
            "you can analyze the common ground among these interactions to determine the user’s preferences " +
            "{preference}. "
            "Then please recommend a suitable item to the user.",
    'out_t': "{item}",
    'input_fields': ['history', 'preference'],
    'output_fields': ['item'],
    'template_id': f"{TradRec_task_key}-{len(TradRec_group)}",
}
TradRec_group[f"{TradRec_task_key}-{len(TradRec_group)}"] = Template(**template)


template = {
    'in_t': "To make a recommendation for this user, we need to analyze their historical interactions, " +
            "which are shown below: {history}. As we know, historical interactions can reflect the user’s preferences. " +
            "Based on this user’s preferences {preference}, " +
            "please recommend an item that you think would be suitable for them.",
    'out_t': "{item}",
    'input_fields': ['history', 'preference'],
    'output_fields': ['item'],
    'template_id': f"{TradRec_task_key}-{len(TradRec_group)}",
}
TradRec_group[f"{TradRec_task_key}-{len(TradRec_group)}"] = Template(**template)


template = {
    'in_t': "The behavioral sequence of the user is shown below: {history}, " +
            "which can be used to infer the user’s preferences {preference}. " +
            "Then please select the item that is likely to be interacted with the user, " +
            "please select one by comparing all items and their similarities to the user’s preference.",
    'out_t': "{item}",
    'input_fields': ['history', 'preference'],
    'output_fields': ['item'],
    'template_id': f"{TradRec_task_key}-{len(TradRec_group)}",
}
TradRec_group[f"{TradRec_task_key}-{len(TradRec_group)}"] = Template(**template)


template = {
    'in_t': "You have observed that the user has clicked on the following items: {history}, " +
            "indicating his personal tastes: {preference} Based on this information, " +
            "please select one item that you think would be suitable for the user.",
    'out_t': "{item}",
    'input_fields': ['history', 'preference'],
    'output_fields': ['item'],
    'template_id': f"{TradRec_task_key}-{len(TradRec_group)}",
}
TradRec_group[f"{TradRec_task_key}-{len(TradRec_group)}"] = Template(**template)


template = {
    'in_t': "You have some information about this user, which is shown below: {preference}, " +
            "the user’s historical interactions: {history} Based on this information, " +
            "please recommend the next possible item for the user, which should match the user’s preference.",
    'out_t': "{item}",
    'input_fields': ['preference', 'history'],
    'output_fields': ['item'],
    'template_id': f"{TradRec_task_key}-{len(TradRec_group)}",
}
TradRec_group[f"{TradRec_task_key}-{len(TradRec_group)}"] = Template(**template)


template = {
    'in_t': "You have obtained the user’s historical interaction list, which is as follows:{history}. " +
            "Based on this history, you can infer the user’s preferences {preference}. " +
            "Now, you need to select the next product to recommend to the user. " +
            "Please choose one item.",
    'out_t': "{item}",
    'input_fields': ['history', 'preference'],
    'output_fields': ['item'],
    'template_id': f"{TradRec_task_key}-{len(TradRec_group)}",
}
TradRec_group[f"{TradRec_task_key}-{len(TradRec_group)}"] = Template(**template)


template = {
    'in_t': "The user has previously purchased the following items: {history}. " +
            "This information indicates their personalized preferences {preference}. " +
            "Based on this information, what will the user likely interact with next?",
    'out_t': "{item}",
    'input_fields': ['history', 'preference'],
    'output_fields': ['item'],
    'template_id': f"{TradRec_task_key}-{len(TradRec_group)}",
}
TradRec_group[f"{TradRec_task_key}-{len(TradRec_group)}"] = Template(**template)


template = {
    'in_t': "Based on the user’s historical interaction list, which is provided as follows: {history} , " +
            "you can infer the user’s personalized preference {preference}. " +
            "And your task is to use this information to predict which item will the user click on next.",
    'out_t': "{item}",
    'input_fields': ['history', 'preference'],
    'output_fields': ['item'],
    'template_id': f"{TradRec_task_key}-{len(TradRec_group)}",
}
TradRec_group[f"{TradRec_task_key}-{len(TradRec_group)}"] = Template(**template)


template = {
    'in_t': "Please recommend an item to the user based on the following information about the user: {preference} , " +
            "the user’s historical interaction, which is as follows: {history} " +
            "Try to select one item that is consistent with the user’s preference.",
    'out_t': "{item}",
    'input_fields': ['preference', 'history'],
    'output_fields': ['item'],
    'template_id': f"{TradRec_task_key}-{len(TradRec_group)}",
}
TradRec_group[f"{TradRec_task_key}-{len(TradRec_group)}"] = Template(**template)


template = {
    'in_t': "You have the ability to infer a user’s preferences based on his past interactions. " +
            "You are provided with a list of the user’s past interactions : {history} " +
            "Your task is to analyze the commonalities among the past interactions and infer his overall preference. " +
            "Please provide your analysis and inference.",
    'out_t': "{preference}",
    'input_fields': ['history'],
    'output_fields': ['preference'],
    'template_id': f"{TradRec_task_key}-{len(TradRec_group)}",
}
TradRec_group[f"{TradRec_task_key}-{len(TradRec_group)}"] = Template(**template)


template = {
    'in_t': "As we all know, the user’s historical interactions are guided by his personalized preference. " +
            "Try to infer the user’s preferences by analyzing his historical interactions: {history}",
    'out_t': "{preference}",
    'input_fields': ['history'],
    'output_fields': ['preference'],
    'template_id': f"{TradRec_task_key}-{len(TradRec_group)}",
}
TradRec_group[f"{TradRec_task_key}-{len(TradRec_group)}"] = Template(**template)


template = {
    'in_t': "You are a recommender system, and are good at recommending products to a user based on his preferences. " +
            "Given the user’s preferences: {preference}, " +
            "please recommend products that are consistent with those preferences.",
    'out_t': "{item}",
    'input_fields': ['preference'],
    'output_fields': ['item'],
    'template_id': f"{TradRec_task_key}-{len(TradRec_group)}",
}
TradRec_group[f"{TradRec_task_key}-{len(TradRec_group)}"] = Template(**template)


template = {
    'in_t': "As we know, a user’s behavior is driven by his preferences, " +
            "which determine what they are likely to buy next. " +
            "Your task is to predict what products a user will purchase next, based on his preferences. " +
            "Given the user’s preferences as follows: {preference}, please make your prediction.",
    'out_t': "{item}",
    'input_fields': ['preference'],
    'output_fields': ['item'],
    'template_id': f"{TradRec_task_key}-{len(TradRec_group)}",
}
TradRec_group[f"{TradRec_task_key}-{len(TradRec_group)}"] = Template(**template)


################################################################################################################
#                                                                                                              #
#                                       product search templates                                               #
#                                                                                                              #
################################################################################################################


ProductSearch_group = {}

template = {
    'in_t': "Suppose you are a search engine, now the user search that {preference/vague_intention/specific_intention}, " +
            "can you generate the item to respond to user’s query?",
    'out_t': "{item}",
    'input_fields': ['preference/vague_intention/specific_intention'],
    'output_fields': ['item'],
    'template_id': f"{ProductSearch_task_key}-{len(ProductSearch_group)}",
}
ProductSearch_group[f"{ProductSearch_task_key}-{len(ProductSearch_group)}"] = Template(**template)


template = {
    'in_t': "As a search engine, your task is to answer the user’s query by generating a related item. " +
            "The user’s query is provided as {specific_intention}. Please provide your generated item as your answer.",
    'out_t': "{item}",
    'input_fields': ['specific_intention'],
    'output_fields': ['item'],
    'template_id': f"{ProductSearch_task_key}-{len(ProductSearch_group)}",
}
ProductSearch_group[f"{ProductSearch_task_key}-{len(ProductSearch_group)}"] = Template(**template)


template = {
    'in_t': "As a recommender system, your task is to recommend an item that is related to the user’s request, " +
            "which is specified as follows: {specific_intention} Please provide your recommendation.",
    'out_t': "{item}",
    'input_fields': ['specific_intention'],
    'output_fields': ['item'],
    'template_id': f"{ProductSearch_task_key}-{len(ProductSearch_group)}",
}
ProductSearch_group[f"{ProductSearch_task_key}-{len(ProductSearch_group)}"] = Template(**template)


template = {
    'in_t': "If a user asks a question like: {preference/vague_intention/specific_intention} " +
            "Please generate a related answer to help him.",
    'out_t': "{item}",
    'input_fields': ['preference/vague_intention/specific_intention'],
    'output_fields': ['item'],
    'template_id': f"{ProductSearch_task_key}-{len(ProductSearch_group)}",
}
ProductSearch_group[f"{ProductSearch_task_key}-{len(ProductSearch_group)}"] = Template(**template)


template = {
    'in_t': "If a user wants to search for something specific in a search engine but doesn’t know how to phrase the query, " +
            "we can help generate the query for them. Now the user wants to search for {item}. " +
            "Please generate the query.",
    'out_t': "{specific_intention}",
    'input_fields': ['item'],
    'output_fields': ['specific_intention'],
    'template_id': f"{ProductSearch_task_key}-{len(ProductSearch_group)}",
}
ProductSearch_group[f"{ProductSearch_task_key}-{len(ProductSearch_group)}"] = Template(**template)


template = {
    'in_t': 'As a search engine that has seen many user queries, ' +
            'you can make an educated guess about how a user might write a query when searching for a particular item. ' +
            'If a user were searching for the item: {item} ' +
            'They might use keywords related to the item such as its brand, or type. So the query would be',
    'out_t': "{specific_intention}",
    'input_fields': ['item'],
    'output_fields': ['specific_intention'],
    'template_id': f"{ProductSearch_task_key}-{len(ProductSearch_group)}",
}
ProductSearch_group[f"{ProductSearch_task_key}-{len(ProductSearch_group)}"] = Template(**template)


template = {
    'in_t': 'You are a search engine and you meet a user’s query {preference/vague_intention/specific_intention}. ' +
            'Please respond to this user by selecting an item.',
    'out_t': "{item}",
    'input_fields': ['preference/vague_intention/specific_intention'],
    'output_fields': ['item'],
    'template_id': f"{ProductSearch_task_key}-{len(ProductSearch_group)}",
}
ProductSearch_group[f"{ProductSearch_task_key}-{len(ProductSearch_group)}"] = Template(**template)


template = {
    'in_t': 'Your task is to select the best itemthat meets the user’s needs ' +
            'based on their search query. ' +
            'Here is the search query of the user: {preference/vague_intention/specific_intention} ',
    'out_t': "{item}",
    'input_fields': ['preference/vague_intention/specific_intention'],
    'output_fields': ['item'],
    'template_id': f"{ProductSearch_task_key}-{len(ProductSearch_group)}",
}
ProductSearch_group[f"{ProductSearch_task_key}-{len(ProductSearch_group)}"] = Template(**template)


template = {
    'in_t': "Your task is to select the best item that matches the user’s query, " +
            "by comparing the items and their relevance to the user’s query. " +
            "The user has entered the following search query: {preference/vague_intention/specific_intention}",
    'out_t': "{item}",
    'input_fields': ['preference/vague_intention/specific_intention'],
    'output_fields': ['item'],
    'template_id': f"{ProductSearch_task_key}-{len(ProductSearch_group)}",
}
ProductSearch_group[f"{ProductSearch_task_key}-{len(ProductSearch_group)}"] = Template(**template)


################################################################################################################
#                                                                                                              #
#                                       personalized search templates                                          #
#                                                                                                              #
################################################################################################################


PersonalizedSearch_group = {}

template = {
    'in_t': "You are a search engine. Here is the historical interaction of a user: {history}. " +
            "And his personalized preferences are as follows: {preference}. " +
            "Your task is to generate a new product that are consistent with the user’s preference.",
    'out_t': "{item}",
    'input_fields': ['history', 'preference'],
    'output_fields': ['item'],
    'template_id': f"{PersonalizedSearch_task_key}-{len(PersonalizedSearch_group)}",
}
PersonalizedSearch_group[f"{PersonalizedSearch_task_key}-{len(PersonalizedSearch_group)}"] = Template(**template)


template = {
    'in_t': "The user has interacted with a list of items, which are as follows: {history}. " +
            "Based on these interacted items, the user current intent are as follows {vague_intention}, " +
            "and your task is to generate products that match the user’s current intent.",
    'out_t': "{item}",
    'input_fields': ['history', 'vague_intention'],
    'output_fields': ['item'],
    'template_id': f"{PersonalizedSearch_task_key}-{len(PersonalizedSearch_group)}",
}
PersonalizedSearch_group[f"{PersonalizedSearch_task_key}-{len(PersonalizedSearch_group)}"] = Template(**template)


template = {
    'in_t': "As a shopping guide, you are assisting a user who has recently purchased the following items: {history} " +
            "The user has expressed a desire for additional products with the following characteristics: {vague_intention} " +
            "Please provide recommendations for products that meet these criteria.",
    'out_t': "{item}",
    'input_fields': ['history', 'vague_intention'],
    'output_fields': ['item'],
    'template_id': f"{PersonalizedSearch_task_key}-{len(PersonalizedSearch_group)}",
}
PersonalizedSearch_group[f"{PersonalizedSearch_task_key}-{len(PersonalizedSearch_group)}"] = Template(**template)


template = {
    'in_t': "As a search engine, you are assisting a user who is searching for the query: {specific_intention}. " +
            "Your task is to recommend products that match the user’s query and also align with their preferences " +
            "based on their historical interactions, which are reflected in the following: {history}",
    'out_t': "{item}",
    'input_fields': ['specific_intention', 'history'],
    'output_fields': ['item'],
    'template_id': f"{PersonalizedSearch_task_key}-{len(PersonalizedSearch_group)}",
}
PersonalizedSearch_group[f"{PersonalizedSearch_task_key}-{len(PersonalizedSearch_group)}"] = Template(**template)


template = {
    'in_t': "The user has recently purchased the following items: {history} " +
            "Now he is interested in finding information about an item that he believe he still need, " +
            "which is: {item}. " +
            "However, the user is unsure how to write a query to search for this item based on their preferences. " +
            "Please assist the user in writing a query.",
    'out_t': "{specific_intention/vague_intention}",
    'input_fields': ['history', 'item'],
    'output_fields': ['specific_intention/vague_intention'],
    'template_id': f"{PersonalizedSearch_task_key}-{len(PersonalizedSearch_group)}",
}
PersonalizedSearch_group[f"{PersonalizedSearch_task_key}-{len(PersonalizedSearch_group)}"] = Template(**template)


template = {
    'in_t': "The user has the following historical interactions: {history}. " +
            "And he is interested in purchasing the target item: {item}. " +
            "Please analyze the user’s historical interactions and identify his preferences that " +
            "lead the user to interact with the target item.",
    'out_t': "{preference}",
    'input_fields': ['history', 'item'],
    'output_fields': ['preference'],
    'template_id': f"{PersonalizedSearch_task_key}-{len(PersonalizedSearch_group)}",
}
PersonalizedSearch_group[f"{PersonalizedSearch_task_key}-{len(PersonalizedSearch_group)}"] = Template(**template)


template = {
    'in_t': "The user has searched for the query: {preference/vague_intention/specific_intention} and " +
            "ultimately selected the product: {item} Based on the user’s query and final choice, " +
            "you can infer their preferences. " +
            "Additionally, the user’s historical interactions have also been influenced by their preferences. " +
            "Please estimate the user’s historical interactions that match their preferences, " +
            "taking into account their search query and final product selection.",
    'out_t': "{history}",
    'input_fields': ['preference/vague_intention/specific_intention', 'item'],
    'output_fields': ['history'],
    'template_id': f"{PersonalizedSearch_task_key}-{len(PersonalizedSearch_group)}",
}
PersonalizedSearch_group[f"{PersonalizedSearch_task_key}-{len(PersonalizedSearch_group)}"] = Template(**template)


template = {
    'in_t': "As a search engine, you have observed the following behavioral sequence of a user: {history}" +
            "Using the content and categories of the user’s historical interactions, " +
            "you can infer their preferences. " +
            "Please make a prediction about the user’s next query and the product he is likely to ultimately purchase, " +
            "based on his preference.",
    'out_t': "{preference/vague_intention/specific_intention} {item}",
    'input_fields': ['history'],
    'output_fields': ['preference/vague_intention/specific_intention', 'item'],
    'template_id': f"{PersonalizedSearch_task_key}-{len(PersonalizedSearch_group)}",
}
PersonalizedSearch_group[f"{PersonalizedSearch_task_key}-{len(PersonalizedSearch_group)}"] = Template(**template)


template = {
    'in_t': "A user’s query can provide insight into their preferences as well as their future purchasing intentions. " +
            "Furthermore, a user’s behavior is often influenced by their preferences. " +
            "Given the user’s query: {preference/vague_intention/specific_intention} " +
            "Please analyze the query to speculate on what products the user has previously purchased and " +
            "predict what products they are likely to purchase next based on their past queries and preferences.",
    'out_t': "{history} {item}",
    'input_fields': ['preference/vague_intention/specific_intention'],
    'output_fields': ['history', 'item'],
    'template_id': f"{PersonalizedSearch_task_key}-{len(PersonalizedSearch_group)}",
}
PersonalizedSearch_group[f"{PersonalizedSearch_task_key}-{len(PersonalizedSearch_group)}"] = Template(**template)


template = {
    'in_t': "Using the user’s current query: {vague_intention/specific_intention}. and " +
            "their historical interactions: {history}. " +
            "You can estimate the user’s preferences {preference}. " +
            "Please respond to the user’s query by selecting an item that best matches their preference and query.",
    'out_t': "{item}",
    'input_fields': ['vague_intention/specific_intention', 'history', 'preference'],
    'output_fields': ['item'],
    'template_id': f"{PersonalizedSearch_task_key}-{len(PersonalizedSearch_group)}",
}
PersonalizedSearch_group[f"{PersonalizedSearch_task_key}-{len(PersonalizedSearch_group)}"] = Template(**template)


template = {
    'in_t': "The user wants to buy some products and searches for: {vague_intention/specific_intention}. " +
            "In addition, they have previously bought: {history}. " +
            "You can estimate their preference by analyzing his historical interactions: {preference}. " +
            "Please recommend one that best matches their search query and preferences.",
    'out_t': "{item}",
    'input_fields': ['vague_intention/specific_intention', 'history', 'preference'],
    'output_fields': ['item'],
    'template_id': f"{PersonalizedSearch_task_key}-{len(PersonalizedSearch_group)}",
}
PersonalizedSearch_group[f"{PersonalizedSearch_task_key}-{len(PersonalizedSearch_group)}"] = Template(**template)


template = {
    'in_t': "A user enjoys shopping very much and has purchased a lot of goods. They are: {history}. " +
            "His historical interactions can reflect his personalized preference. {preference}. " +
            "Now he wants to buy some new items, such as: ’{vague_intention/specific_intention}’ " +
            "Please select the item that best meets the user’s needs and preferences.",
    'out_t': "{item}",
    'input_fields': ['history', 'preference', 'vague_intention/specific_intention'],
    'output_fields': ['item'],
    'template_id': f"{PersonalizedSearch_task_key}-{len(PersonalizedSearch_group)}",
}
PersonalizedSearch_group[f"{PersonalizedSearch_task_key}-{len(PersonalizedSearch_group)}"] = Template(**template)


template = {
    'in_t': "Based on the user’s historical interactions with the following items: {history}. " +
            "You can infer his preference by analyzing the historical interactions. {preference} " +
            "Now the user wants to buy a new item and searches for: “{vague_intention/specific_intention}” " +
            "Please select a suitable item that matches his preference and search intent.",
    'out_t': "{item}",
    'input_fields': ['history', 'preference', 'vague_intention/specific_intention'],
    'output_fields': ['item'],
    'template_id': f"{PersonalizedSearch_task_key}-{len(PersonalizedSearch_group)}",
}
PersonalizedSearch_group[f"{PersonalizedSearch_task_key}-{len(PersonalizedSearch_group)}"] = Template(**template)


TaskTemplate = {
    "TradRec": TradRec_group,
    'ProductSearch': ProductSearch_group,
    'PersonalizedSearch': PersonalizedSearch_group,
}
TaskNum = {
    "TradRec": 1,
    'ProductSearch': 1,
    'PersonalizedSearch': 2,
}


ValTradRec_group = {}
template = {
    'in_t': "Base on user’s historical interactions: {history}" +
            "you need to choose the item, which should match the user’s preference, " +
            "among the following candidates: {candidate_items}",
    'out_t': "{item}",
    'input_fields': ['history', 'candidate_items'],
    'output_fields': ['item'],
    'template_id': f"{ValTradRec_task_key}-{len(ValTradRec_group)}",
}
ValTradRec_group[f"{ValTradRec_task_key}-{len(ValTradRec_group)}"] = Template(**template)
ValTaskTemplate = {
    "ValTradRec": ValTradRec_group
}
ValTaskNum = {
    "ValTradRec": 1,
}

ValFullRec_group = {}
template = {
    'in_t': "Base on user’s historical interactions: {history}" +
            "you need to choose the item, which should match the user’s preference.",
    'out_t': "{item}",
    'input_fields': ['history'],
    'output_fields': ['item'],
    'template_id': f"{ValFullRec_task_key}-{len(ValFullRec_group)}",
}
ValFullRec_group[f"{ValFullRec_task_key}-{len(ValFullRec_group)}"] = Template(**template)

TestTradRec_group = {}
template = {
    'in_t': "Base on user’s historical interactions: {history}" +
            "you need to choose the item, which should match the user’s preference, " +
            "among the following candidates: {candidate_items}",
    'out_t': "{item}",
    'input_fields': ['history', 'candidate_items'],
    'output_fields': ['item'],
    'template_id': f"{TestTradRec_task_key}-{len(TestTradRec_group)}",
}
TestTradRec_group[f"{TestTradRec_task_key}-{len(TestTradRec_group)}"] = Template(**template)
TestTaskTemplate = {
    "TestTradRec": TestTradRec_group
}
TestTaskNum = {
    "TestTradRec": 1,
}

TestFullRec_group = {}
template = {
    'in_t': "Base on user’s historical interactions: {history}" +
            "you need to choose the item, which should match the user’s preference.",
    'out_t': "{item}",
    'input_fields': ['history'],
    'output_fields': ['item'],
    'template_id': f"{TestFullRec_task_key}-{len(TestFullRec_group)}",
}
TestFullRec_group[f"{TestFullRec_task_key}-{len(TestFullRec_group)}"] = Template(**template)

Intention_plus_group = {}
template = {
    'in_t': "I like {category} products.",
    'out_t': "",
    'input_fields': ['category'],
    'output_fields': [],
    'template_id': f"IntentionPlus-{len(Intention_plus_group)}",
}
Intention_plus_group[f"IntentionPlus-{len(Intention_plus_group)}"] = Template(**template)

template = {
    'in_t': "Please recommend some {category} items.",
    'out_t': "",
    'input_fields': ['category'],
    'output_fields': [],
    'template_id': f"IntentionPlus-{len(Intention_plus_group)}",
}
Intention_plus_group[f"IntentionPlus-{len(Intention_plus_group)}"] = Template(**template)

template = {
    'in_t': "I'm interested in {category}.",
    'out_t': "",
    'input_fields': ['category'],
    'output_fields': [],
    'template_id': f"IntentionPlus-{len(Intention_plus_group)}",
}
Intention_plus_group[f"IntentionPlus-{len(Intention_plus_group)}"] = Template(**template)

template = {
    'in_t': "I would like to buy some {category} products",
    'out_t': "",
    'input_fields': ['category'],
    'output_fields': [],
    'template_id': f"IntentionPlus-{len(Intention_plus_group)}",
}
Intention_plus_group[f"IntentionPlus-{len(Intention_plus_group)}"] = Template(**template)

template = {
    'in_t': "I would like to browse some {category} products",
    'out_t': "",
    'input_fields': ['category'],
    'output_fields': [],
    'template_id': f"IntentionPlus-{len(Intention_plus_group)}",
}
Intention_plus_group[f"IntentionPlus-{len(Intention_plus_group)}"] = Template(**template)

template = {
    'in_t': "I prefer in {category} items",
    'out_t': "",
    'input_fields': ['category'],
    'output_fields': [],
    'template_id': f"IntentionPlus-{len(Intention_plus_group)}",
}
Intention_plus_group[f"IntentionPlus-{len(Intention_plus_group)}"] = Template(**template)


Intention_minus_group = {}
template = {
    'in_t': "I don't like {category} products.",
    'out_t': "",
    'input_fields': ['category'],
    'output_fields': [],
    'template_id': f"IntentionMinus-{len(Intention_minus_group)}",
}
Intention_minus_group[f"IntentionMinus-{len(Intention_minus_group)}"] = Template(**template)

template = {
    'in_t': "Please exclude any {category} item.",
    'out_t': "",
    'input_fields': ['category'],
    'output_fields': [],
    'template_id': f"IntentionMinus-{len(Intention_minus_group)}",
}
Intention_minus_group[f"IntentionMinus-{len(Intention_minus_group)}"] = Template(**template)

template = {
    'in_t': "I'm not interested in {category}.",
    'out_t': "",
    'input_fields': ['category'],
    'output_fields': [],
    'template_id': f"IntentionMinus-{len(Intention_minus_group)}",
}
Intention_minus_group[f"IntentionMinus-{len(Intention_minus_group)}"] = Template(**template)

template = {
    'in_t': "Don't recommend me any {category} products",
    'out_t': "",
    'input_fields': ['category'],
    'output_fields': [],
    'template_id': f"IntentionMinus-{len(Intention_minus_group)}",
}
Intention_minus_group[f"IntentionMinus-{len(Intention_minus_group)}"] = Template(**template)

template = {
    'in_t': "I don't want to browse any {category} product",
    'out_t': "",
    'input_fields': ['category'],
    'output_fields': [],
    'template_id': f"IntentionMinus-{len(Intention_minus_group)}",
}
Intention_minus_group[f"IntentionMinus-{len(Intention_minus_group)}"] = Template(**template)

template = {
    'in_t': "I hate {category} items",
    'out_t': "",
    'input_fields': ['category'],
    'output_fields': [],
    'template_id': f"IntentionMinus-{len(Intention_minus_group)}",
}
Intention_minus_group[f"IntentionMinus-{len(Intention_minus_group)}"] = Template(**template)