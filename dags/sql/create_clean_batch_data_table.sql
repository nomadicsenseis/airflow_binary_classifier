CREATE TABLE IF NOT EXISTS clean_batch_data (
    Customer_ID SERIAL PRIMARY KEY,
    rev_Mean FLOAT,
    mou_Mean FLOAT,
    totmrc_Mean FLOAT,
    da_Mean FLOAT,
    ovrmou_Mean FLOAT,
    ovrrev_Mean FLOAT,
    vceovr_Mean FLOAT,
    datovr_Mean FLOAT,
    roam_Mean FLOAT,
    change_mou FLOAT,
    change_rev FLOAT,
    drop_vce_Mean FLOAT,
    drop_dat_Mean FLOAT,
    blck_vce_Mean FLOAT,
    blck_dat_Mean FLOAT,
    unan_vce_Mean FLOAT,
    unan_dat_Mean FLOAT,
    plcd_vce_Mean FLOAT,
    plcd_dat_Mean FLOAT,
    recv_vce_Mean FLOAT,
    recv_sms_Mean FLOAT,
    comp_vce_Mean FLOAT,
    comp_dat_Mean FLOAT,
    custcare_Mean FLOAT,
    ccrndmou_Mean FLOAT,
    cc_mou_Mean FLOAT,
    inonemin_Mean FLOAT,
    threeway_Mean FLOAT,
    mou_cvce_Mean FLOAT,
    mou_cdat_Mean FLOAT,
    mou_rvce_Mean FLOAT,
    owylis_vce_Mean FLOAT,
    mouowylisv_Mean FLOAT,
    iwylis_vce_Mean FLOAT,
    mouiwylisv_Mean FLOAT,
    peak_vce_Mean FLOAT,
    peak_dat_Mean FLOAT,
    mou_peav_Mean FLOAT,
    mou_pead_Mean FLOAT,
    opk_vce_Mean FLOAT,
    opk_dat_Mean FLOAT,
    mou_opkv_Mean FLOAT,
    mou_opkd_Mean FLOAT,
    drop_blk_Mean FLOAT,
    attempt_Mean FLOAT,
    complete_Mean FLOAT,
    callfwdv_Mean FLOAT,
    callwait_Mean FLOAT,
    churn FLOAT,
    months FLOAT,
    uniqsubs FLOAT,
    actvsubs FLOAT,
    totcalls FLOAT,
    totmou FLOAT,
    totrev FLOAT,
    adjrev FLOAT,
    adjmou FLOAT,
    adjqty FLOAT,
    avgrev FLOAT,
    avgmou FLOAT,
    avgqty FLOAT,
    avg3mou FLOAT,
    avg3qty FLOAT,
    avg3rev FLOAT,
    avg6mou FLOAT,
    avg6qty FLOAT,
    avg6rev FLOAT,
    hnd_price FLOAT,
    phones FLOAT,
    models FLOAT,
    truck FLOAT,
    rv FLOAT,
    lor FLOAT,
    adults FLOAT,
    income FLOAT,
    numbcars FLOAT,
    forgntvl FLOAT,
    eqpdays FLOAT,
    new_cell_N smallint,
    new_cell_U smallint,
    new_cell_Y smallint,
    crclscod_A smallint,
    crclscod_A2 smallint,
    crclscod_AA smallint,
    crclscod_B smallint,
    crclscod_B2 smallint,
    crclscod_BA smallint,
    crclscod_C smallint,
    crclscod_C2 smallint,
    crclscod_C5 smallint,
    crclscod_CA smallint,
    crclscod_CC smallint,
    crclscod_D smallint,
    crclscod_D4 smallint,
    crclscod_D5 smallint,
    crclscod_DA smallint,
    crclscod_E smallint,
    crclscod_E4 smallint,
    crclscod_EA smallint,
    crclscod_EM smallint,
    crclscod_G smallint,
    crclscod_GA smallint,
    crclscod_H smallint,
    crclscod_I smallint,
    crclscod_J smallint,
    crclscod_JF smallint,
    crclscod_K smallint,
    crclscod_M smallint,
    crclscod_O smallint,
    crclscod_TP smallint,
    crclscod_U smallint,
    crclscod_U1 smallint,
    crclscod_V smallint,
    crclscod_W smallint,
    crclscod_Y smallint,
    crclscod_Z smallint,
    crclscod_Z1 smallint,
    crclscod_Z4 smallint,
    crclscod_ZA smallint,
    asl_flag_N smallint,
    asl_flag_Y smallint,
    prizm_social_one_C smallint,
    prizm_social_one_R smallint,
    prizm_social_one_S smallint,
    prizm_social_one_T smallint,
    prizm_social_one_U smallint,
    area_ATLANTIC_SOUTH_AREA smallint,
    area_CALIFORNIA_NORTH_AREA smallint,
    area_CENTRAL_SOUTH_TEXAS_AREA smallint,
    area_CHICAGO_AREA smallint,
    area_DALLAS_AREA smallint,
    area_DC_MARYLAND_VIRGINIA_AREA smallint,
    area_GREAT_LAKES_AREA smallint,
    area_HOUSTON_AREA smallint,
    area_MIDWEST_AREA smallint,
    area_NEW_ENGLAND_AREA smallint,
    area_NEW_YORK_CITY_AREA smallint,
    area_NORTH_FLORIDA_AREA smallint,
    area_NORTHWEST_ROCKY_MOUNTAIN_AREA smallint,
    area_OHIO_AREA smallint,
    area_PHILADELPHIA_AREA smallint,
    area_SOUTH_FLORIDA_AREA smallint,
    area_SOUTHWEST_AREA smallint,
    area_TENNESSEE_AREA smallint,
    dualband_N smallint,
    dualband_T smallint,
    dualband_U smallint,
    dualband_Y smallint,
    refurb_new_N smallint,
    refurb_new_R smallint,
    hnd_webcap_UNKW smallint,
    hnd_webcap_WC smallint,
    hnd_webcap_WCMB smallint,
    ownrent_O smallint,
    ownrent_R smallint,
    dwlltype_M smallint,
    dwlltype_S smallint,
    marital_A smallint,
    marital_B smallint,
    marital_M smallint,
    marital_S smallint,
    marital_U smallint,
    infobase_M smallint,
    HHstatin_A smallint,
    HHstatin_B smallint,
    HHstatin_C smallint,
    HHstatin_G smallint,
    HHstatin_H smallint,
    HHstatin_I smallint,
    dwllsize_A smallint,
    dwllsize_B smallint,
    dwllsize_C smallint,
    dwllsize_D smallint,
    dwllsize_E smallint,
    dwllsize_F smallint,
    dwllsize_G smallint,
    dwllsize_H smallint,
    dwllsize_I smallint,
    dwllsize_J smallint,
    dwllsize_K smallint,
    dwllsize_L smallint,
    dwllsize_M smallint,
    dwllsize_N smallint,
    dwllsize_O smallint,
    ethnic_B smallint,
    ethnic_D smallint,
    ethnic_F smallint,
    ethnic_G smallint,
    ethnic_H smallint,
    ethnic_I smallint,
    ethnic_J smallint,
    ethnic_M smallint,
    ethnic_N smallint,
    ethnic_O smallint,
    ethnic_P smallint,
    ethnic_R smallint,
    ethnic_S smallint,
    ethnic_U smallint,
    ethnic_X smallint,
    ethnic_Z smallint,
    kid0_2_U smallint,
    kid0_2_Y smallint,
    kid3_5_U smallint,
    kid3_5_Y smallint,
    kid6_10_U smallint,
    kid6_10_Y smallint,
    kid11_15_U smallint,
    kid11_15_Y smallint,
    kid16_17_U smallint,
    kid16_17_Y smallint,
    creditcd_N smallint,
    creditcd_Y smallint
);
