CDF       
      obs    @   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       ?Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM????   
add_offset               min       ?h?t?j~?   max       ?Õ?$?/        ?   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M??   max       P??        ?   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ????   max       =??        ?   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ??z?G?{   
add_offset               min       @?
=p??
   max       @E?ffffg     
    ?   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ??z?G?{   
add_offset        @f?        min       ??\(?    max       @vx(?\     
   *?   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ??         
add_offset               min       @&         max       @P?           ?  4?   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @?F        max       @?P             5,   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ??C?   max       >\(?        6,   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A?KO   max       B,vZ        7,   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A?S-   max       B,?#        8,   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?{!q   max       C?}?        9,   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?p?   max       C?i?        :,   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          ?        ;,   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          9        <,   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          '        =,   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M??   max       P?        >,   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6??C-   
add_offset               min       ??֡a??f   max       ????s?        ?,   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ????   max       =??#        @,   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ??z?G?{   
add_offset               min       @?\(?   max       @E??\)     
   A,   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ??z?G?{   
add_offset        @f?        min       ??         max       @vw??Q?     
   K,   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ??         
add_offset               min       @%         max       @P?           ?  U,   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @?F        max       @??@            U?   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         C?   max         C?        V?   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6??C-   
add_offset               min       ??Ov_ح?   max       ??-V        W?   
             
                  
   	   +   	            n   /   >      	      )      Z      %                  9      /   $         "   
            '                      ]               ?      ?               N}}?NW8_N???O??|N?~yN??\O?*M??P??N3'O1-5N?caP2'O?ONVpBO??<O???P??OX7P=G?N'?1OT?N?o?O??N?o?P??N86rOS?UO???N?ȳN??`N0??Om?3P1?O"?sO?]O?=?O<?N?h?O???N?P?OS?:N`??NMOϼ?O\ukOv?O?|NNUZ?OO%?OpӼP)??OE؎Nx??O"O+??P{?O?rO?nSNΨ;N_??N{	?Og?O?l???????
?????e`B?49X?#?
?ě??o??o;??
;?`B<#?
<49X<D??<D??<e`B<e`B<u<?C?<?C?<?C?<?t?<?t?<???<??
<??
<?1<?9X<?j<?j<?j<ě?<ě?<???<???<???<???<?`B<?`B=\)=\)=\)=t?=??=?w=#?
='??='??='??='??=,1=0 ?=0 ?=49X=49X=@?=P?`=Y?=?%=?+=??-=??w=?^5=??nmty????????wtnnnnnn????????????????????")69<:<6)( [Y[cdnz?????????zna[
#'0<<CC><0#
???????????????????????????????????????? ??	        5/1:HTYYamz????j[TH5[\gghtuttg[[[[[[[[[[???????????????????? "(/:;HSQPHD;/)"    ???
/DUafcH/#??#)25;>@A<55)%????????????????????? )04NQOB5)?????
#).,&#
????????(+?LNKB)?????63367BO[ajqtxrg[OB:6).Mht?????????t[O@6)[hmttu????th[[[[[[[[??????
$'#
???,/+/<>HNPHAHHH<63/,,????????????????????)5<A5,)'??????$#%('???	
	
#&'$# 
						 #/<HU_^VUHC<7/%#ry????????????????zr??????
#/15/,#
???:;<@HQUZ_UTUaUNHC<::??????????????????????????????????BT[gvdg[N5???qptv??????????????tq????????????????????0/16BO[hlz}tph[OB;60????????


		????????????????????')5<@EFD;;5)$EDKNR[gitg[NEEEEEEEE??????	
#(&%!
?????
 !###
???????????????????????????)3BLS[fgd[OD6)|z{???????????????|????????????????????????????
!??????

???????????????????RMLLNPTamz|zz~xmaTR????????????????????????????????????????????????????????????zvw???????????????zz???
#&4630)#
??|????"/0??????|????????? ')????????????

?????_bin{|??????????{nb_??

?????????? !????????????????????????????????????????ÓàèìöìãàÖÓÇÃÇËÓÓÓÓÓÓ???????????????????????????????????????нսݽ????ݽнĽ????????Ľʽннннн?E?E?E?E?FFFFFFE?E?E?E?E?E?E?E?E?Eͻ??????????????????????x?u?n?p?v?x???????B?N?[?^?^?[?Z?N?B?5?)?(?)?0?5?>?B?B?B?B?ûлԻԻڻֻܻû???????????????????????ìù????????????ÿùìæìììììììì?5?A?N?Z?a?p???????????????Z?N?A?5?%?'?5????????????????????????????????????????????????????????????????ŹŭůŹż???????a?d?m?s?z?{??z?m?a?W?T?J?P?T?\?a?a?a?a?m?????????y?Z?T?G?;?,?????.?;?T?`?m?ʾ׾??????????????׾ʾ??????????????ž?????$?%???????????????????????????????????u?m?`???7?8???G?T?m???????????&?/?)???? ?????????????????0?I?U?e?w??{?n?I?#?
????????????????0?????ʼӼּ????????ּʼ??????????????????????ʾ;ƾ????????????s?b?a?H?G?Q?f?z???4?<?@?M?M?M?E?@?4?0?1?0?4?4?4?4?4?4?4?4?Z?f?s?????????????????s?f?Z?A?B?F?M?ZEEE*E3E,E*E#EEE	ED?D?D?D?D?EE	EE?r??????????ͼммԼʼ????????r?f?T?V?r?Z?f?k?m?q?f?Z?P?M?L?M?T?Z?Z?Z?Z?Z?Z?Z?Z??-?S?_?t????l?_?S?F?:?!?????????????h?tāăā?w?|?t?h?[?Z?[?a?f?h?h?h?h?h?h???????????????????????????????????"?/?5?6?9?7?1?/?"??	????????????	??;???H?M?T?Y?e?e?e?b?a?T?I?H?B?E?F?;?9?;ÇÓØàèìïìàÕÓÒÇ?~?z?y?zÂÇÇ?????????????????????????????????????????"?.?G?T?`?y???{?y?m?_?G?;?.?'?????"?5?N?W?Q?5?"?(?>?<???? ?????׿޿???5?A?M?W?Z?_?f?j?e?f?n?f?Z?M?A?4?.?/?4?>?A???'?3?@?D?K?Q?M?3?'???????ҹҹܹ???l?x?????????~?q?l?S?F?;?-?'?)?.?7?F?_?l?/?<?H?L?U?Y?a?c?a?U?P?H?@?<?/?#??#?#?/???????????????????????z?z??????Ƴ???????????????????ƧƁ?h?M?U?\ƎƙƳčĚĦĩĦģěĚčăĂĆčččččččč?A?M?Z?f?s?~??????????s?f?M?A?;?7?6?9?A¦§²½¸²¦¥¦¦¦¦¦¦ù????????úùìåìîòùùùùùùùù?e?~???????ɺݺֺɺĺ??????????r?]?U?Y?e?)?6?B?O?W?[?f?a?[?S?B?6?)?????? ?)?????Ŀѿݿ???? ?????ݿѿ?????????????????(?A?N?g?????????????s?g?Z?A?1?????C?O?\?h?r?h?\?Z?O?C?9?>?C?C?C?C?C?C?C?C?????????????????????y?l?i?d?Z?`?l?y?????I?U?b?n?{ŇŒŜşŚŔŇ?{?b?U?I?C?B?E?Iù??????????????????àÇ?^?Z?d?àìù?O?[?h?t?~??x?t?h?[?O?B?6?'??)?3?6?B?O?????????????????????????????????????????/?<?H?Q?U?_?[?U?H?C?<?/?#?#???#?(?/?/????!?#?,?.?2?.?!??????????????????'?@?I?D?N?Q?O?@?'????л??????м?!? ?'¿?????????
???%??????????????¹¹¿DoD{D?D?D?D?D?D?D?D?D?D?D?D{DhDYDWDVDbDo??????????!?'?(?!???????????????ǭǭǪǢǡǔǈǇǈǏǓǔǟǡǢǭǭǭǭǭ?????????????????????????????????????????A?N?Z?a?g?l?h?g?a?`?Z?N?I?:?5?2?1?5?@?A???????????????????????????????????????? ) P .  N I 1 o e - o < N L ) I Q .  - ? 3 N = F 5 P # R ? B z I d / E A 5 + n N A I ^ K & O m m 6 B M T K ( ! R ? 6  b E 3 d  |  ?  ?  ?  /  ?  >  C  ?  6  ?  ?  ?  H  m    ?  #  ?  "  ?  ?  ?  ?  ?  [  q  ?  x  ?  ?  Q    ?  h  ?  T  3  ?  ?  ?  ?  ?  Q  ?  ?  	  ?  m  ?  ?    ?  ?  #  h  ?  ?  ?  =  ?  ?  @  ^??C??e`B?e`B<?C??o??o<?9X:?o<??
<49X<?o<???=Y?<?1<?C?=0 ?<???>   =?o=??w<?9X<???<???=q??<???=?G?<?/=q??=8Q?<?/<?`B<?/='??=??T=o=?t?=y?#=?w=<j=?C?=49X=m?h=D??=H?9=??-=?t?=y?#=?\)=@?=y?#=m?h>C?=}??=@?=?\)=?o>>v?=??T>\(?=??-=??
=?{=???=??#B
(?BqBBn?Bz?B%X?B?B"?B??A?s?B	C?B*?A?KOBĤB??BV?B??BwB?>B??B??B?B??B??B"}BƳBw?BQ4B?lB??B??Bl?B=?Bw?B?rB
?%B?CB??B?FBD?B??B?B$2?BKB!??B??B0vB??B??B?IB,vZA??B{B
?B?8B??B$?BРB?B?B(??B??B?B9?BwBB
?DBA?BIBHB%I?B??B"<DB??A?D7B	A)B?A?S-B??B??BV6B??B@?B?#B?B??B??B?$B?B!??BڡBD?B? B?B??B?ZBF0BȪB?B?pB
?[B?B?eB B??B~1B?-B$@B1B!??BG
B@?B?@B?B?B,?#A?s'B?yB?#B??B??B$?%B?7B.VB:?B(N~B??B??B90B?GA??A???A(GbC?}?@??jA???@?v?A???A?=A??A???A??'Ai3dAQR?AԓeAl??A??|A???@???AG?@Έ?AB?C?ri@?/A??@x?A?A?t?A???A?;A?{ Ba?Ad??A?IA<m??{!q@?@?A???AHN?B?IA??A??A??QA?%@??A??NAy??A?<?Bq?A|?A???A?u?AٿdA???A?TtA	f@???A??vC??6@_??Bn9A?>mA???A??Aʓ?A???A(??C?i?@?qA??@?A?~A??A?v?A?i?A??XAj?AR1?A?I/Aj?A?[A??Y@???AG??@??A@?C?s?@?E?A?$?@w??Aۆ?AҀA?,&A?tA?	?B?eAcO2A??A<?g?p?@?%A??AHΫB?wA?A??A???A͂?@m?A׀&Aw7?A??LBqA??A?ouA?u?A?w?A??A?|4AjH@?IvA???C?ȑ@T
Bw_A???A??A?:
   
             
                  
   
   +   
            o   0   >      	      )      [      &                  9      0   $         "   
            '   !                  ^               ?      ?                                          +            -         %      3      +            #      %                        9      !            +               %         '            +               7   #   !                                                      !         #      !      #                                          #                                 !         %            %               '                     N}}?NW8_N???O???N?~yNR?|Oh<%M??Opt
N3'O1-5N???O?G?O?ONVpBO??AO???O?a?OL??O???N'?1OT?N[?~OX??Nv?O/:yN86rN瑼O{?*N?ȳN??`N0??O/??O??$N?{PON??OA?SO<?N?h?O?ƓN?P?O+??N`??NMO?UO\ukOv?O?N&NUZ?N?JOpӼP?OE؎Nx??N???O+??P?OT??OF?NΨ;N_??N{	?Og?O?l  ?  ?  ?  ?    ?  c  ?  J  G  H      G  c  ?  ?  	?  	_    ?  ?  L    +  
?  ?  ?  ?  >  ?  L  h  ?  ?  ?  ?  ?  ?    ?  b  ?       ?  ?  ?  t  ?  B  
?  q    ?  4    ?  ?  ?  ?  ?  ?  ????????
?????49X?49X?t???o?o;ě?;??
;?`B<49X<?9X<D??<D??<?o<e`B=?+<???=t?<?C?<?t?<??
=o<?9X=?+<?1=+<???<?j<?j<ě?<?/=?w<?/=?P=C?<?`B<?`B=49X=\)=??=t?=??=<j=#?
='??=0 ?='??=P?`=,1=aG?=0 ?=49X=T??=@?=?Q?=q??=??#=?+=??-=??w=?^5=??nmty????????wtnnnnnn????????????????????")69<:<6)( ]\agnz??????????zna]
#'0<<CC><0#
???????????????????????????????????????? ??	        ZWZ__agmpuz??????zaZ[\gghtuttg[[[[[[[[[[????????????????????"")/;=HOMNHA;/+"""""????
#/>KSL</#
??#)25;>@A<55)%????????????????????)2BHPNB5)?????
#).,&#
?????????)1574)???55;BMO[elqsmh`[OB=85A=>@FO[ht??????th[OA[hmttu????th[[[[[[[[??????
$'#
???//1/,/<<HMNH<<://///????????????????????)56<5)???????????	
	
#&'$# 
						&%%//<@HMUUURHD<1/&&}u}????????????????}??????
#/15/,#
???:;<@HQUZ_UTUaUNHC<::????????????????????????????
?????????"4BB:5)???tt~???????????xttttt????????????????????@=:88BO[hhnsojh[OHB@????????


		??????????????????)5;?@@=5)#EDKNR[gitg[NEEEEEEEE??????
"##!
?????
 !###
???????????????????????????):BIOW__VFA6)|z{???????????????|??????????????????????????
???????

????????????????????RMLLNPTamz|zz~xmaTR???????????????????????????????????????????????????????????????????????????????????
#&4630)#
??????? ??????????????  ???????????

????_bin{|??????????{nb_??

?????????? !????????????????????????????????????????ÓàèìöìãàÖÓÇÃÇËÓÓÓÓÓÓ???????????????????????????????????????нսݽ????ݽнĽ????????Ľʽннннн?E?E?E?FFFFFFFE?E?E?E?E?E?E?E?E?Eͻ??????????????????????x?u?n?p?v?x???????B?N?Z?Y?N?M?B?5?4?2?5?@?B?B?B?B?B?B?B?B?ûͻϻֻ׻лû?????????????????????????ìù????????????ÿùìæìììììììì?N?Z?g?????????????????????s?g?Z?W?P?K?N????????????????????????????????????????????????????????????????ŹŭůŹż???????a?b?m?r?y?z?~?z?m?a?Z?T?N?R?T?`?a?a?a?a?m?y???????????????y?`?G?;?3?&?2?G?V?f?m?ʾ׾??????????????׾ʾ??????????????ž?????$?%????????????????????????????????y?m?`?C?9?;?A?T?`?m?????????????&?/?)???? ??????????????????#?0?<?D?M?P?L???0?#???????????????????ʼּ޼??????ּʼ????????????????????????????????????????????????f?X?V?W?e??4?<?@?M?M?M?E?@?4?0?1?0?4?4?4?4?4?4?4?4?Z?f?s?????????????????s?f?Z?A?B?F?M?ZEEEE!E*E1E*E*E!EEEED?EEEEEE?r??????????????????????????~?r?c?c?m?r?Z?f?g?i?g?f?Z?T?P?X?Z?Z?Z?Z?Z?Z?Z?Z?Z?Z?!?-?:?F?R?S?^?\?S?F?:?-?!???????!?h?tāăā?w?|?t?h?[?Z?[?a?f?h?h?h?h?h?h??????????	????????????????????????	??"?/?3?4?6?6?4?/?"???	???????????	?;???H?M?T?Y?e?e?e?b?a?T?I?H?B?E?F?;?9?;ÇÓØàèìïìàÕÓÒÇ?~?z?y?zÂÇÇ?????????????????????????????????????????.?;?G?T?Z?`?m?v?m?k?`?V?G?;?.?"???"?.??(?5?B?H?E?5?(????????????????????M?M?Z?[?^?\?Z?M?A?4?4?3?4?<?A?M?M?M?M?M??????'?3?<?A?@???3?'????????????????F?S?_?l?x?????x?l?l?_?S?F?:?5?0?5?:?A?F?/?<?H?L?U?Y?a?c?a?U?P?H?@?<?/?#??#?#?/???????????????????????z?z??????????????????????????ƳƧƐƅƎƔƚƦƳ??čĚĦĩĦģěĚčăĂĆčččččččč?A?M?Z?f?s?x???????s?f?Z?M?A???:?9?<?A¦§²½¸²¦¥¦¦¦¦¦¦ù????????úùìåìîòùùùùùùùù?~???????ɺӺɺú????????~?r?b?[?`?e?r?~?)?6?B?O?W?[?f?a?[?S?B?6?)?????? ?)?????Ŀѿݿ???? ?????ݿѿ????????????????(?A?N?d?????????????s?g?Z?I?3?)?"???(?C?O?\?h?r?h?\?Z?O?C?9?>?C?C?C?C?C?C?C?C?y??????????????????y?t?l?h?l?y?y?y?y?y?I?U?b?n?{ŇŒŜşŚŔŇ?{?b?U?I?C?B?E?Iìù???????????????????Ñ?z?n?n?xÇÓì?O?[?h?t?~??x?t?h?[?O?B?6?'??)?3?6?B?O?????????????????????????????????????????/?<?H?S?O?H?>?<?/?*?$?&?/?/?/?/?/?/?/?/????!?#?,?.?2?.?!???????????????????'?4?<?C?C?=?4?'?????ջɻʻѻۻ??????????
??????
????????????????????D{D?D?D?D?D?D?D?D?D?D?D?D?D?D{DoDkDiDoD{??????????!?'?(?!???????????????ǭǭǪǢǡǔǈǇǈǏǓǔǟǡǢǭǭǭǭǭ?????????????????????????????????????????A?N?Z?a?g?l?h?g?a?`?Z?N?I?:?5?2?1?5?@?A???????????????????????????????????????? ) P .  N ? , o [ - o = K L ) L Q   % ? 3 A 5 M * P  E ? B z > P 8 B 5 5 + 5 N @ I ^ I & O f m 3 B A T K - ! ! $ 2  b E 3 d  |  ?  ?  [  /  ^  ?  C  ?  6  ?  ?  ?  H  m  ?  ?  ?  ?  ?  ?  ?  n  ?  J  p  q    
  ?  ?  Q  ~    ?  ?  ?  3  ?    ?  u  ?  Q  ?  ?  	  Q  m  ?  ?  y  ?  ?  ?  h  _  ?  ?  =  ?  ?  @  ^  C?  C?  C?  C?  C?  C?  C?  C?  C?  C?  C?  C?  C?  C?  C?  C?  C?  C?  C?  C?  C?  C?  C?  C?  C?  C?  C?  C?  C?  C?  C?  C?  C?  C?  C?  C?  C?  C?  C?  C?  C?  C?  C?  C?  C?  C?  C?  C?  C?  C?  C?  C?  C?  C?  C?  C?  C?  C?  C?  C?  C?  C?  C?  C?  ?  ?  ?  ?  ?  ?  ?  ?  ?  x  h  X  F  3      ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  |  o  b  Q  <  (    ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  o  ]  J  7  $    t    ?  u  h  [  S  Z  d  k  e  V  :    ?  Y  ?    ?   ?        ?  ?  ?  ?  ?  ?  ?  ?  j  M  .    ?  ?  ?  z  R  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  s  \  D  -     ?   ?  /  M  `  c  a  X  K  9  #    ?  ?  r  2  ?  ?  "  ?  ?  ?  ?  ?  ~  v  n  g  _  W  O  F  >  6  -  $      ?  ?  ?  ?  ?  ?  ?      )  G  G  =  +      ?  ?  ?  ?  ?  p  <  ?  G  >  5  ,  "        ?  ?  ?  ?  ?  ?  ?  ?  ?  o  ]  J  H  =  2  "    ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ]  ?  ?  a          	  ?  ?  ?  ?  ?  ?  ?  ?  ?  l  I  #  ?  ?  k  ?  ?  ?  ?  ?      ?  ?  ?  ?  \    ?  O    ?  ?    ?  G  B  >  5  ,      ?  ?  ?  ?  ?  ?  m  P  .  ?  ?  r  "  c  d  e  f  g  f  f  e  e  e  e  e  c  a  _  ]  h  v  ?  ?  ?  ?  ?  ?  ?  ?  ?  u  R  0      ?  ?  d  %  ?  ?    ?  ?  x  h  S  ;  "    ?  ?  ?  f  5    ?  ?  ?  `  B     ?  ?  F  ?  ?  	  	A  	k  	?  	?  	?  	?  	V  	  ?  ,  ?  ?  ?  /  ?  	  	:  	S  	^  	Z  	E  	  ?  ?  ~  7  ?  q  ?  ^  ?    S  ?  ?     j  ?  ?  ?        ?  ?  ?  m  $  ?  }    ?  
  b  ?  ?  ?  ?  ?  ?  ?  ?  ?  z  x  u  s  ?  ?  ?  %  B  [  s  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  |  l  b  c  d      '  O  s  z  o  e  Z  N  @  ,    ?  ?  ?  ?  ?  ?  ?  V  s  w  t  v  {    |  s  e  S  8    ?  ?  Q  	  ?  /  ?      #  &  )  *  *  +  (  !        ?  ?  ?  ?  l  E    ?  w  		  	?  	?  
9  
m  
?  
?  
?  
?  
?  
?  
K  	?  	.  R  E  ?  %  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  x  r  o  l  i  i  i  i  ?  ?  +  a  ?  ?  ?  ?  ?  y  X  -     ?  d  ?  ,  I  `  s  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  m  U  ;     ?  ?  ?  u  C  ?  >  6  .  &             ?  ?  ?  ?  ?  ?  ?  ?  e  K  0  ?  ?  ?  ?  x  d  O  ;  (      ?  ?  ?  ?  ?  ?  {  a  G  L  B  8  .  $        ?  ?  ?  ?  ?  ?  ?  U  %   ?   ?   ?  Q  W  d  h  g  c  [  M  <  '    ?  ?  ?  ?  ?  ?  q  K      ?  O  v  ?  ?  ?  z  c  @    ?  ?  >  ?  g  ?    ?   ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  y  r  k  g  i  k  u  ?  ?    G  ?  ?  ?  ?  ?  ?  ?  ?  ?  n  B  ?  |  ?  .  Y  A    ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  h  F    ?  ?    1  ?  ?  ?  ?  w  d  Q  9       ?  ?  ?  ?  ?  ?  p  R  4  $    ?  ?  ?  ?  ?  ?  ?  k  O  /    ?  ?  Z  (  ?  ?  f    ?  ?  ?  ?  ?  ?        ?  ?  ?  ?  ?  a  &  ?  ?  O  ?  y  ?  ?  ?  {  h  T  ?  *    ?  ?  ?  ?  ?  ?  r  Z  D  :  0  M  Y  `  b  _  W  K  :    ?  ?  ?    M    ?  ?  T  ?  ?  ?  "  I  Z  l  }  ?  ?  ?  ?  ?  ?  ?  ?  ?  m  G    ?  ?         ?  ?  ?  ?  ?  v  H    ?  ?  ?  _  ,  ?  ?  ?  X  ?            ?  ?  ?  ?  b  (  ?  ?  l  3  ?  {    6  ?  ?  ?  |  w  n  b  O  2    ?  ?  ?  `  "  ?  w    w  8  ?  ?  ?  s  Z  G  0    ?  ?  ?  ?  n  D    ?  z    ~   ?  ?  ?  ?  d  A    ?  ?  ?  `  B  3  ?  ?  Z    ?  B  ?  {  t  l  d  \  U  N  H  A  ;  5  5  ;  A  B  8  /  %        ?  ?    1  C  V  g  v  ?  ?  ?  ~  `  4  ?  ?  ~  =  ?  ?  B  +    ?  ?  ?  ?  ?  ?  ?  ?  z  V  ,  ?  ?  Q  ?  ?   ?  
#  
?  
?  
?  
?  
?  
?  
?  
?  
m  
K  
  	?  	?  	    ?  ?  ?  ?  q  e  U  A  .  (        ?  ?  ?  ~  Q  !  ?  ?  z  (  ?                              ?  ?  ?  ?  s  W  ?  '  c  ?  ?  ?  ?  ?  ?  ?  ?  +  ?  j    ?    ?  ?  g  4  !          ?  ?  ?  ?  ?  ?  ?  ?  ?  \  7    ?  ?  
?  m  ?  ?        	  ?  ?  _  
?  
?  
Y  	?  	W  h    P  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  a  :  
  ?  ?  <    a  ?  I  ?    o  ?  ?  ?  ?  ?  N  ?  ?  ?  h    ?  ?  u  
  ?  ?  ?  ?  ?  ?  i  M  -    ?  ?  ?  ?  y  E    ?  ?  =  ?  ?  ?  ?  ?  ?  ?  ?  k  V  @  +      ?  ?  ?  ?  ?  ?  ?  ?  ?  x  b  L  5      ?  ?  ?  ?  b  <    ?  ?  {  ?  ?  ?  ?  ?  ?  u  ]  :    ?  ?  i  &  ?  n  ?  ?    ?  '  ?  ?  ?  ?  ?  ?  W  '  ?  ?  ?  T    ?  i    ?  $  ?  #