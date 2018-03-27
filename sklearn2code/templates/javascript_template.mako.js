function isNan(val) {
  if (val !== val) {
    return true;
  } else {
    return false;
  }
};

function expit(val) {
	if x >= 0:
        z = Math.exp(-val)
        return 1 / (1 + z)
    else:
        z = Math.exp(val)
        return z / (1 + z)
}

%for function_ in functions:
function ${namer(function_)}(${', '.join(map(str, function_.inputs)) + ', ' if function_.inputs else ''}) {
	%for assignments, (called_function, arguments) in function_.calls:
	var [${', '.join(map(str, assignments))}] = ${namer(called_function)}(${', '.join(map(str, arguments))})
	%endfor
	return [${', '.join(map(printer, function_.outputs))}]
}
%endfor
