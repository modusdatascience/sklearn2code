<%!
from sklearn2code.sym.function import VariableNameFactory
%>

function expit(val) {
	if (val >= 0) {
        var z = Math.exp(-val);
        return 1 / (1 + z);
	} else {
        var z = Math.exp(val);
        return z / (1 + z);
	};
};

function weightedMode(data, weights) {
	var counts = new Map(data.map(x => [x, 0]))
	for (i=0; i<data.length; i++){
		counts[data[i]] += weights[i];
	};
	var best_key;
	var best_val;
	var first = true;
	var val;
	var keys = Array.from(counts.keys());
	keys.sort(function(a, b){return a - b});
	keys.reverse();
	for (var key of keys) {
		val = counts[key];
		if (first) {
			best_key = key;
			best_val = val;
		}
		if (val > best_val) {
			best_val = val;
			best_key = key;
		}
	}
	return best_key
}

function weightedMedian(data, weights) {
	var order = data.map((x, i) => [x, i]).sort((a, b) => a[0] - b[0]).map(x => x[1]);
	var total = 0.0;
	var half = weights.reduce((a, b) => a + b, 0) / 2.0;
	var i;
	for (i=0; i<order.length; i++) {
		total += weights[order[i]];
		if (total > half) {
			break;
		};
	};
	return data[order[i]];
}

function assign(arr, vars) {
    var x = {};
    var num = Math.min(arr.length, vars.length);
    for (var i = 0; i < num; ++i) {
        x[vars[i]] = arr[i];
    }
    return x;
}

%for function_ in functions:
	<%
	Var = VariableNameFactory(existing=function_.all_variables())
	%>
function ${namer(function_)}(${', '.join(map(str, function_.inputs))}) {
	%for assignments, (called_function, arguments) in function_.calls:
	<%
	dummy = Var()
	%>
	var ${dummy} = assign(${namer(called_function)}(${', '.join(map(str, arguments))}), [${', '.join(map(lambda x: '"%s"' % str(x), assignments))}])
//	var [${', '.join(map(str, assignments))}] = ${namer(called_function)}(${', '.join(map(str, arguments))});
	%for assgn in assignments:
	${str(assgn)} = ${dummy}.${str(assgn)}
	%endfor
	%endfor
	return [${', '.join(map(printer, function_.outputs))}];
};
%endfor
