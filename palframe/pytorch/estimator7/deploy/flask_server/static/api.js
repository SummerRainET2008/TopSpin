// 获取模型池的状态
function get_model_pool_status(options){
    console.log('正在获取模型池状态')
    return $.ajax({
            url: '/status',
            type: 'GET',
         })
}


// 增加实例
function add_one_instance(options){
    console.log('正在获取增加模型池实例',options)
    return $.ajax({
        url: '/add_one_instance',
        contentType: "application/json",
        type: 'POST',
        dataType: "json",
        data: JSON.stringify(options)
    })

}


// 删除实例
function delete_one_instance(options){
    console.log('正在获取删除模型池实例',options)
    return $.ajax({
        url: '/delete_one_instance',
        contentType: "application/json",
        type: 'POST',
        dataType: "json",
        data: JSON.stringify(options)
    })

}

// 提交任务
function submit_task(options){
    console.log('正在获取删除模型池实例',options)
    return $.ajax({
        url: '/submit_task',
        contentType: "application/json",
        type: 'POST',
        dataType: "json",
        data: JSON.stringify(options)
    })
}


