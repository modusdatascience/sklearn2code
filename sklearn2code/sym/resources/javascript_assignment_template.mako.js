function ${function_name}(${', '.join(input_names)}) {
    var [${', '.join(symbols)}] = ${function_name}(${', '.join(input_symbols)})
%for line in body_code.splitlines():
    ${line};
%endfor
    ${return_code};
};
