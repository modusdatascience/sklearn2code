from sklearn2code.sym.expression import Expression



class VectorExpression(Expression):
    def __init__(self, dim):
        self.dim = int(dim)
        if self.dim != dim:
            raise ValueError('The dim of a VectorExpression must be an integer.')
    
    











# from ..utility import tupify
# from frozendict import frozendict
# from .expression import Expression
# 
# 
# class ParameterizedTypeFactoryType(object):
#     def __init__(self, base_types):
#         self.base_types = tupify(base_types)
# #         self.type_parameter_names = type_parameter_names
#         self.specializations = {}
#     
#     def __call__(self, name, **type_parameters):
#         type_parameters = frozendict(type_parameters)
#         if type_parameters not in self.specializations:
#             result = type(name, self.base_types, {})
#             for param_name, param in type_parameters.items():
#                 setattr(result, param_name, param)
#             type_parameters = tupify(type_parameters)
#             self.specializations[type_parameters] = result
#         else:
#             result = self.specializations[type_parameters]
#         return result 
# 
# class VectorBase(Expression):
#     pass


