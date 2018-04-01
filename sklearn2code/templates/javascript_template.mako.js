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

%for function_ in functions:
function ${namer(function_)}(${', '.join(map(str, function_.inputs))}) {
	%for assignments, (called_function, arguments) in function_.calls:
	var [${', '.join(map(str, assignments))}] = ${namer(called_function)}(${', '.join(map(str, arguments))});
	%endfor
	return [${', '.join(map(printer, function_.outputs))}];
};
%endfor
