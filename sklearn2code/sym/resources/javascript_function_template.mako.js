function ${function_name}(${', '.join(input_names)}) {
%for line in body_code.splitlines():
    ${line};
%endfor
    ${return_code};
};
