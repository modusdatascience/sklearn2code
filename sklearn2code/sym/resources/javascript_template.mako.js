function missing(val) {
  if (val !== val) {
    return 1;
  } else {
    return 0;
  }
};

function nanprotect(val) {
  if (val !== val) {
    return 0;
  } else {
    return val;
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

%for function_code in functions:
${function_code}
%endfor
