from pyearth.earth import Earth
from ..base import syms
from ..function import Function, tupify
from ..base import sym_predict, sym_transform, input_size
from toolz.functoolz import identity
from ..expression import RealVariable as Symbol, Sum as Add, Product as Mul, Max, RealNumber, \
        Piecewise, Power as Pow, And, nan, Not, IsNan
from pyearth._basis import LinearBasisFunction, HingeBasisFunction, SmoothedHingeBasisFunction, \
          MissingnessBasisFunction, ConstantBasisFunction, VariableBasisFunction
from sklearn2code.sym.expression import Integer, BooleanExpression, BoolToReal,\
    true
from operator import __add__
from six.moves import reduce

def export_sympy_term_expressions(earth_model):
    """
    Construct a list of expressions for all non-pruned terms in the model.

    :param earth_model: Fitted pyearth model
    :return: a list of expressions representing terms in the model.  These
      expressions are the symbolic equivalent of the Earth.transform method.

    """
    
    Missing = IsNan
    NaNProtect = lambda x: Piecewise((RealNumber(0), IsNan(x)), (x, true))

    def linear_bf_to_factor(bf, bf_var):
        return bf_var

    def smoothed_hinge_bf_to_factor(bf, bf_var):
        knot = RealNumber(bf.get_knot())
        knot_minus = RealNumber(bf.get_knot_minus())
        knot_plus = RealNumber(bf.get_knot_plus())
        r = RealNumber(bf.get_r())
        p = RealNumber(bf.get_p())
        if bf.get_reverse():
            lower_p = (-(bf_var - knot)), (bf_var <= knot_minus)
            upper_p = (RealNumber(0), bf_var >= knot_plus)
            left_exp = Mul(p, Pow((bf_var - knot_plus), Integer(2)))
            right_exp = Mul(r, Pow((bf_var - knot_plus), Integer(3)))
            middle_b = And(knot_minus < bf_var, bf_var < knot_plus)
            middle_exp = (Add(left_exp, right_exp), middle_b)
            piecewise = Piecewise(lower_p, upper_p, middle_exp)
            factor = piecewise
        else:
            lower_p = (RealNumber(0), bf_var <= knot_minus)
            upper_p = (bf_var - knot, bf_var >= knot_plus)
            left_exp = Mul(p, Pow((bf_var - knot_minus), Integer(2)))
            right_exp = Mul(r, Pow((bf_var - knot_minus), Integer(3)))
            middle_b = And(knot_minus < bf_var, bf_var < knot_plus)
            middle_exp = (Add(left_exp, right_exp), middle_b)
            piecewise = Piecewise(lower_p, upper_p, middle_exp)
            factor = piecewise
        return factor

    def hinge_bf_to_factor(bf, bf_var):
        knot = bf.get_knot()
        if bf.get_reverse():
            factor = Max(RealNumber(0), RealNumber(knot) - bf_var)
        else:
            factor = Max(RealNumber(0), bf_var - RealNumber(knot))
        return factor

    def missingness_bf_to_factor(bf, bf_var):
        # This is the error that should be raised when a user attempts to use functionality
        # that has not yet been implemented.
        if bf.complement:
            return Not(Missing(bf_var))
        else:
            return Missing(bf_var)

    def constant_bf_to_factor(bf, bf_var):
        return RealNumber(1)

    def protect_from_nan(label, missables):
        return NaNProtect(Symbol(label)) if label in missables else Symbol(label)

    def dont_protect_from_nan(label, missables):
        return Symbol(label)

    bf_to_factor_dispatcher = {LinearBasisFunction: linear_bf_to_factor,
                               SmoothedHingeBasisFunction: smoothed_hinge_bf_to_factor,
                               HingeBasisFunction: hinge_bf_to_factor,
                               MissingnessBasisFunction: missingness_bf_to_factor,
                               ConstantBasisFunction: constant_bf_to_factor}

    nan_protect_dispatch = {LinearBasisFunction: protect_from_nan,
                            SmoothedHingeBasisFunction: protect_from_nan,
                            HingeBasisFunction: protect_from_nan,
                            MissingnessBasisFunction: dont_protect_from_nan,
                            ConstantBasisFunction: protect_from_nan}

    def bf_to_factor(bf, missables):
        '''
        Convert a BasisFunction to a factor of a term.
        '''
        if isinstance(bf, VariableBasisFunction):
            bf_var = nan_protect_dispatch[bf.__class__](bf.label, missables)

        else:
            bf_var = None
        return bf_to_factor_dispatcher[bf.__class__](bf, bf_var)

    def missingness_bf_get_missables(bf):
        bf_var = bf.label
        return set([bf_var])

    def non_missable(bf):
        return set()

    bf_get_missables_dispatcher = {LinearBasisFunction: non_missable,
                                   SmoothedHingeBasisFunction: non_missable,
                                   HingeBasisFunction: non_missable,
                                   MissingnessBasisFunction: missingness_bf_get_missables,
                                   ConstantBasisFunction: non_missable}

    def get_missables(bf):
        missables = bf_get_missables_dispatcher[bf.__class__](bf)
        parent = bf.get_parent()
        if parent is None:
            return missables
        else:
            missables.update(get_missables(parent))

        return missables

    def bf_to_term(bf, missables):
        '''
        Convert a BasisFunction to a term (without coefficient).
        '''
        def make_real(arg):
            if isinstance(arg, BooleanExpression):
                return BoolToReal(arg)
            else:
                return arg
        term = bf_to_factor(bf, missables)
        parent = bf.get_parent()
        if parent is None:
            return term
        else:
            try:
                return Mul(make_real(term), bf_to_term(parent, missables))
            except:
                return Mul(make_real(term), bf_to_term(parent, missables))

    return [bf_to_term(bf, get_missables(bf)) for bf in earth_model.basis_.piter()]


def export_sympy(earth_model, columns=None):
    """
    Constructs a sympy expression or list of sympy expressions from of a trained earth model.

    :param earth_model: Trained pyearth model
    :param columns: The index or indices of the output columns for which expressions are to
      be constructed.  If an integer is used, a sympy expression is returned.  If indices
      are given then a list of sympy expressions is returned.  If columns is None, it is treated
      as if columns=0 for models with only one output column or as columns=slice(None) for more than
      one output column.
    :return: a sympy expression or list of sympy expressions equivalent to the Earth.predict method for
      the selected output columns.

    """
    # Set a sane default for columns
    if columns is None:
        if earth_model.coef_.shape[0] == 1:
            columns = 0
        else:
            columns = slice(None)

    # Get basis function terms
    terms = export_sympy_term_expressions(earth_model)

    # Handle column choice
    coefs = earth_model.coef_[columns]
    if len(coefs.shape) == 1:
        unwrap = True
        coefs = [coefs]
        n_cols = 1
    else:
        unwrap = False
        n_cols = coefs.shape[0]

    # Combine coefficients with terms for each output column
    result = [reduce(__add__, [RealNumber(coefs[i][j]) * term for j, term in enumerate(terms)], RealNumber(0)) for i in range(n_cols)]

    if unwrap:
        # Result should be an expression rather than a list of expressions.
        result = result[0]
    return result

@input_size.register(Earth)
def input_size_earth(estimator):
    return len(estimator.xlabels_)

@syms.register(Earth)
def syms_earth(estimator):
    return [Symbol(label) for label in estimator.xlabels_]

@sym_transform.register(Earth)
def sym_transform_earth(estimator):
    inputs = syms(estimator)
    calls = tuple()
    outputs = tuple(export_sympy_term_expressions(estimator))
    return Function(inputs, calls, outputs)

@sym_predict.register(Earth)
def sym_predict_earth(estimator):
    inputs = syms(estimator)
    calls = tuple()
    outputs = tupify(export_sympy(estimator))
    return Function(inputs, calls, outputs)
