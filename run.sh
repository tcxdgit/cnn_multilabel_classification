#!/bin/bash
IFS=$'\n\n'
#field=$1
#action=$2
source ../function.sh

function choose_action(){
    option=$(whiptail --title "命令选择" --clear --menu "请选择其中一个：" 12 35 3 \
         "test" "聊天测试" "find" "查找进程" "kill" "关闭进程" "restart" "重启进程" \
        3>&1 1>&2 2>&3)
    exitstatus=$?
    if [ $exitstatus = 0 ]; then
        echo $option
    else
        echo ''
    fi
}

function kill_process(){
    echo 'kill process'
    ids=$(get_pid 'mq_cnnclassify' ' '${field}' ')
    echo "kill mq_cnnclassify ${field}"
    
    if [[ ${ids} != '' ]]; then
        for id in ${ids}
        do
            echo 'kill '${id}
            kill ${id}
        done
    fi
}

field=$(choose_field)

if [[ ${field} = 'bank_psbc' ]];then
    field='psbc'
fi

if [[ ${field} != '' ]]; then
    action=$(choose_action)
fi

if [[ ${action} != '' ]]; then
    if [[ ${action} = 'restart' || ${action} = 'start' ]]; then
        num=$(input_num)
        if [[ ${num} != '' ]]; then
            confirm=$(choose_confirm '场景：'$field ' 命令：'$action'\n数量：'$num)
        fi
    else
        confirm=$(choose_confirm '场景：'$field' 命令：'$action' ')
    fi
#    confirm=$(choose_confirm '场景：'$field' 命令：'$action' ')
fi

if [[ ${confirm} != '' ]]; then
    case $action in
        'test')
            echo ${field}
            python3.5 classify.py ${field}
        ;;
        'find')
            result=$(get_pid 'mq_cnnclassify' ' '${field}' ')
            echo 'mq_classify: '${result}
        ;;
        'kill')
            result=$(kill_process)
            for line in ${result}
            do
                echo $line
            done
        ;;
        'restart')
            result=$(kill_process)
            for line in ${result}
            do
                echo $line
            done
            echo 'start process'

            if [ ${field} = 'psbc' ];then
                path="../work_space/categories/module/cnn_multi"
            else
                path="../work_space/${field}/module/cnn_multi"
            fi
            echo ${path}

            for i in `seq $num`
            do
                nohup python3.5 mq_cnnclassify.py ${field} ${path} >/dev/null 2>nohup.out &
            done
            echo 'done!'
        ;;
    esac
fi
