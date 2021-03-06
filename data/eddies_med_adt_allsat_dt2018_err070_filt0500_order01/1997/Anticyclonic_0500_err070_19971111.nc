CDF       
      obs    :   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       ?Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM????   
add_offset               min       ?h?t?j~?   max       ??\(??      ?  ?   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N
?9   max       P???      ?  ?   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ?D??   max       =??m      ?  |   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ??z?G?{   
add_offset               min       @>?G?z?   max       @E??z?H     	   d   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ??z?G?{   
add_offset        @f?        min       ???
=p?    max       @vu?????     	  )t   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ??         
add_offset               min       @1?        max       @R            t  2?   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @?x        max       @?P@          ?  2?   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ????   max       >^5?      ?  3?   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A?ZH   max       B,?      ?  4?   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A???   max       B,??      ?  5?   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       @#۩   max       C?~?      ?  6?   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       @!	%   max       C?h?      ?  7?   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          ?      ?  8h   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ;      ?  9P   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          -      ?  :8   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N
?9   max       PLܟ      ?  ;    speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6??C-   
add_offset               min       ??D??*1   max       ??????      ?  <   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ?49X   max       =??m      ?  <?   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ??z?G?{   
add_offset               min       @>?G?z?   max       @E??z?H     	  =?   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ??z?G?{   
add_offset        @f?        min       @ z?G?    max       @vu?????     	  F?   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ??         
add_offset               min       @)         max       @R            t  O?   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @?x        max       @?9`          ?  Pl   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         DI   max         DI      ?  QT   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6??C-   
add_offset               min       ?o?䎊q?   max       ??L?_?        R<                                    N   &   O                           )         '         &               0   ?   !   +      ;   d            ,               j                  2   '   &   @O?xNϢ?N9sN?	2NK+9O?3O9?N?0?O\?INN??Nw?.P???O?a?PG-HN?)NI??O??O0?NC?O? jN,?N3b PE/?O?h?N?pO?=?O?[O_?
O?k O?ANpl?O?'?N?3?O??rP??+O??O?OV?P@?PPsN
?9N?5?O5PLܟO)^6N??Os?N??#O???N? O6(?NC?O!?}ND? Oj??N?T?O??*O??2?D????/??C??ě??D????o;o;D??;?o;?`B<t?<#?
<49X<D??<D??<e`B<u<u<?t?<?t?<?t?<??
<?j<???<?<???=C?=C?=\)=?P=?w='??=8Q?=@?=D??=D??=D??=D??=H?9=e`B=ix?=m?h=q??=q??=u=u=?%=?+=?+=?\)=??=??=??=??=???=?/=?;d=??mefgimqt?????????tmgekn??????????????ztnk?}|?????????????????EEN[gkttvutpgc[WONEE #+08<FD<0%#        ????????

	??????????????????????????c^hlt?????????}tohcctnmtt??????????????t"/3/,"/.6BOTSOFB>6////////?????
/SnnhH4#
????#!$%/<HUacgb_aUH<.'#?????BHEEB5)?????????????????????????????????????????????????????????????????????????????????????????????????????????

???????#0<>@<20-#`[dgt|yttg``````````\^}???????????????t\??????	?????????????????????????;FSS[m????????zaTH;;???????????????????????? )57:975)?4::=BHUafnz??}znUH<4#'029<<940#????????????????????????????
#(,-*#??????? )/,)????????????	???????????)G[g???gYB5??????????????????????????6BGJK?:6)????????????????????????????5>A=5)????????
#9>=7&#
????










	
#$#!#
				tst}??????????????xtusy??????????????ztu
	)-/2552-)}|????????????}}}}}}????????????????????^\abmz?????zma^^^^^^????????

????????
"#((#
??????????????

?????~?????????????????? ???')2/)& ?????????????????????????????

???^YUV[anvz{}|zwna^^^^jhbinz??????????zpjvrontz?????????????v?H?U?X?a?n?z?Â?z?n?a?U?P?H?<?<?8?<?@?HE?E?E?E?E?E?FFFE?E?E?E?E?E?E?E?E?E?Eټ??????????????z????????????????!?????????????????????????????????????????????????????????????????????????? ?????????ߺ???????????????????????????????????????????Ҽ????????????????????????????????????????N?Z?g?s?t?????????????g?Z?N?A?6?7?>?D?N?H?T?a?b?g?j?a?W?T?J?H?@?H?H?H?H?H?H?H?H?ʼּ????޼ּʼƼ??????¼ʼʼʼʼʼʼʼʿ?.?T?m?????????`?;?.?"????Ҿ??׾?????????????????????????ùííó???????#?I?U?g?o?s?n?\?U?<?????????????????
?#?a?n?zÇÌÇÇ?z?v?n?a?\?a?a?a?a?a?a?a?a???????????????????????????????????????ƻ?????'?6?>?4?'????????????????????ûл׻ۻػллû????????????????????û?ÓÕàãààÓÍÇ?ÇÎÓÓÓÓÓÓÓÓ?r?????????????????????w?r?h?Y?P?Y?`?r?????????????{?r?p?r?{?????????y?????????????y?y?w?y?y?y?y?y?y?y?y?y?y?A?M??????????????????s?=?/?$? ? ?%?0?A?ݿ????????????ݿѿĿ??????????ĿϿ??#?#?/?<?H?K?H?G?<?1?/?*?#??????!?#ĚĦĿ??????????????ĳĦĚĕĔĚėđēĚ?????????????????????????i?g?c?d?g?s?????????????	????	???????????????????????a?z???????????????????????z?w?o?f?`?\?a??????????!?-?!????????????޼׼ܼ??
???????
??????
?
?
?
?
?
?
?
???	??"?/?;???I?P?S?S?H?;?/?"??	???????M?Z?f?h?i?j?g?f?`?Z?M?I?D?F?J?K?M?M?M?M??????????????????????????????????????????)?6?D?K?E?O?a?^?6?)?#???????????????)?6?7?=?B?F?B?7?5?/?)???
?????(?)?ɺкֺ޺ںֺϺ̺ƺ????????~?y?w???????ɽ????????????????????????????????|?v?x????????????#?/?*???????????³«¦²¸???)?B?[?e?t?~?z?g?N?B?)????????????)?M?Z?f?g?f?Z?V?M?A?4?*?4?A?B?M?M?M?M?M?M?????????ɾʾҾʾ???????????????????????????????(?+?(?$?????????????????????????????????????????Z?N?A?6?9?I?g????ƚƧƳ??????????????ƳƩƤƚƎƈƊƎƙƚ?????ÿĿſĿ????????????????????????????????????????????????z?u?m?i?k?m?z?|????Ź????????????????ŹŵůŰŷŹŹŹŹŹŹD?D?D?D?D?D?D?D?D?D?D?D?D?D?D}DvDlDoD?D?ǡǤǬǪǡǠǔǈǃǅǈǌǔǠǡǡǡǡǡǡ???*?@?C?R?\?h?x?u?h?\?O?C?8?*? ????Z?g?s?t?w?s?g?Z?V?N?H?L?N?Z?Z?Z?Z?Z?Z?Z?????????ͻлܻ߻߻ܻۻлû??????????????l?x?????????????x?w?l?c?l?l?l?l?l?l?l?lEuE?E?E?E?E?E?E?E?E?E?E?E?E?E?EuEkEiEiEu?????ûлܻ????ܻԻлû??????????????????ɺֺ????????ݺ׺ɺ????????????????????ɼM?Y?f?r????????w?f?M?@?4?'?#? ?%?'?4?M @ ? 9 C F ; 4 t A R , ) * ) | j j ' 2 > Z = A C > A . . ; # m L d ] ] @ X @ # ( ? H 2 0 P / 6 1 F R v z W H ; - K 1  3  h  B    q  N  V    ?  w  ?  _  7  Y  ^  ?  H  <  8    q  R  m  7  ?    y  ?  I  H  ?  q  ;  ?  ?  .  ?  ?  9  ~  W  ?  ?  ?  |  ?  A  ?  k  ?  ?  o  ?  \  ?    h  ????????
?T??;ě?;o<u<?1<T??<?C?<49X<?9X=?-=D??=?^5<u<???<?`B=+<ě?=\)<?9X<ě?=?%=H?9='??=?O?=ix?=H?9=??=aG?=49X=??=e`B=??>^5?=??T=?^5=?o=?"?>?w=y?#=?+=?7L=???=?hs=?hs=??-=???>0 ?=???=?^5=???=?^5=?`B>?>?+>?+>?|?B	?~B??B)w?B??B%ӻB#F?B?B?(Ba?A?ZHB-.B??B?rB??BdB?,B"2?B#?B ??B"??B%??B	lCB??Ba?B?~A?dqB?B?BkB%<?B??B?B?^B^?B}/B? B??B*:B?	B??B$?B$?uB
??B?B_?Bc?B&?A??B?BcBgBFdBe?B,?B-?B?B??BPB
3oB6?B)??BљB%͋B#<B?_B??B?A???B?$B??B??B?}BHHB?%B"= B#;?B ??B"?B%ŹB	J?B?BEfB??A??:B?B??B?iB%??B?ZBN?B?_B??BF$B?#B?OB)??B?~B??B$?QB$A(B
??B??B?BF_B?A?gVB?tB	?B?B;#B@0B,??BEoB?
BAB??A?LC?~?@?s?A???@??	@Q?AAХ@??A?M?A???A Ab?A??OA?pA?GA?Q@?X?@??7Aʀ?@?@?@??8AoFA@ʩA~??A?Q?A???A?Z A?A???A&?A?ޙA?d?A>|sA?&?A?B?A?|?@#۩A 
?A?d^A??A<??AL??A??YA???B<#Aua?A??OA?A?C?߬Bk?B ??A??@?2?@?k?C?	?@?9@/@?pAƀ?C?h?@?OA??*@??|@Sk_AЅ?@???A???A??@???Af??Aє*A??VA?i4A??h@??@? ?AʃH@??/@? ?An??AB?*AA]A??BA?{?A?wA???A?RA???A?v?A?AA??4A?r?Aօ1@!	%A ??A???A??	A:?rAL?FA??A?jB??AuvA?sA?|?C??Bq?B 2?A?|@??@??gC??@???@*??@?F]                                    N   &   P                           )         '         '               0   ?   "   ,      ;   e      	   	   -               k                  3   (   '   A                                    9      -                           -         #                           ;      !      -   '            -                                                                              -      !                           #                                                               -                                          Nȗ?NϢ?N9sN?	2NK+9O'N_?N?0?O\?INN??NBgP9AOQOO??MN?)NI??N???O	6NC?O&?N,?N3b O??O?N?pO#m?O?\XO_?
O(uPO #tNpl?O?'?N?IO???O?KN ?O?ՅN?ӄO??O???N
?9N?5?O5PLܟO)^6N??Os?N??#O)?N? O6(?NC?O!?}ND? O>J?N?T?O??*O??2    ?  
      X  2  ?  D  ?  /  ?  ?  ?  ?  ?  !  6  ?  Q  ?  U    [  ?  ?  ?  ?  ?  "  ?  ?  ?  e  %  g  ?    :  ?  ?  ?    ?  7  ?  |  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  4?49X??/??C??ě??D??%   <#?
;D??;?o;?`B<49X<?<?t?=??<D??<e`B<?o<?o<?t?<?j<?t?<??
=C?=C?<?=H?9=t?=C?=<j=?w=?w='??=@?=L??=?=?%=P?`=Y?=???=?-=ix?=m?h=q??=q??=u=u=?%=?+=???=?\)=??=??=??=??=?G?=?/=?;d=??mkiknst?????????tkkkkkn??????????????ztnk?}|?????????????????EEN[gkttvutpgc[WONEE #+08<FD<0%#        ????????

??????????????????????????c^hlt?????????}tohcctnmtt??????????????t"/3/,"116BOROOBBB611111111?????
/J_aYG</#
??,'%'+/<HN\_\XWUOH</,?????5<=<95)????????????????????????????????????????????????????????????????????????????????????????????????????????
?????????#0<>@<20-#`[dgt|yttg``````````qr????????????????zq???????????????????????????????????``bfmvz????????zmha`???????????????????????? )57:975)?B?BGHTUagntz|zsnaUHB##$007;820#????????????????????????????
#(,-*#??????),))?????????????????????)BKLKG@5)???????????????????????????7BHGD=86)???????????????????????????????????????
#.10.+#
???










	
#$#!#
				tst}??????????????xtusy??????????????ztu
	)-/2552-)}|????????????}}}}}}????????????????????^\abmz?????zma^^^^^^????????

????????
"#((#
??????????????

?????~?????????????????? ???')2/)& ????????????????????????????

????^YUV[anvz{}|zwna^^^^jhbinz??????????zpjvrontz?????????????v?H?U?a?n?z?|??z?n?c?a?]?U?H?A?@?H?H?H?HE?E?E?E?E?E?FFFE?E?E?E?E?E?E?E?E?E?Eټ??????????????z????????????????!????????????????????????????????????????????????????????????????????????????????????ߺ????????????????????????????????????????????????Ҽ????????????????????????????????????????N?Z?g?s?t?????????????g?Z?N?A?6?7?>?D?N?H?T?a?b?g?j?a?W?T?J?H?@?H?H?H?H?H?H?H?H?ʼּݼ??ּܼʼɼ??????ļʼʼʼʼʼʼʼʿ?.?I?T?m??????`?;?.?"??	??????????????????????????????????þù÷û?????#?0?I?U?]?_?_?[?I?0?#?
?????????????#?a?n?zÇÌÇÇ?z?v?n?a?\?a?a?a?a?a?a?a?a???????????????????????????????????????ƻ?????'?4?4?;?4?'??????????????????ûлԻڻ׻лϻû????????????????????û?ÓÕàãààÓÍÇ?ÇÎÓÓÓÓÓÓÓÓ?r???????????????????{?t?r?f?`?^?f?h?r?????????????{?r?p?r?{?????????y?????????????y?y?w?y?y?y?y?y?y?y?y?y?y?M?Z????????????????s?f?Z?R?A?1?2?5?A?M?ѿݿ????????????????ݿ׿ѿ˿ͿϿѿ??#?#?/?<?H?K?H?G?<?1?/?*?#??????!?#ĳĿ??????????????????ĿĳħĦĢĤĦİĳ???????????????????????????p?g?e?h?t?????????????	????	???????????????????????z???????????????????????????z?v?r?o?o?z??????????!?"?!????????????ټ߼????
???????
??????
?
?
?
?
?
?
?
???	??"?/?;???I?P?S?S?H?;?/?"??	???????Z?`?f?g?h?f?f?]?Z?M?M?G?I?M?M?T?Z?Z?Z?Z??????????????????????????????????????????)?2?5?6?3?)?????????????????????)?3?6?<?6?+?)?????(?)?)?)?)?)?)?)?)???ɺֺۺ׺ͺɺº????????~?{?y??????????????????????????????????????????????????????????
?????
??????????????????????)?5?B?T?a?h?i?f?[?N?B?5?)???????M?Z?f?g?f?Z?V?M?A?4?*?4?A?B?M?M?M?M?M?M?????????ɾʾҾʾ???????????????????????????????(?+?(?$?????????????????????????????????????????Z?N?A?6?9?I?g????ƚƧƳ??????????????ƳƩƤƚƎƈƊƎƙƚ?????ÿĿſĿ????????????????????????????????????????????????z?u?m?i?k?m?z?|????Ź????????????????ŹŵůŰŷŹŹŹŹŹŹD?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?ǡǤǬǪǡǠǔǈǃǅǈǌǔǠǡǡǡǡǡǡ???*?@?C?R?\?h?x?u?h?\?O?C?8?*? ????Z?g?s?t?w?s?g?Z?V?N?H?L?N?Z?Z?Z?Z?Z?Z?Z?????????ͻлܻ߻߻ܻۻлû??????????????l?x?????????????x?w?l?c?l?l?l?l?l?l?l?lEuE?E?E?E?E?E?E?E?E?E?E?E?E?EuEnElEmEqEu?????ûлܻ????ܻԻлû??????????????????ɺֺ????????ݺ׺ɺ????????????????????ɼM?Y?f?r????????w?f?M?@?4?'?#? ?%?'?4?M ? ? 9 C F 7 4 t A R 0 ( ) . | j i ( 2 @ Z = &  > + $ . $ " m L b S % N R >  % ? H 2 0 P / 6 1 ' R v z W H + - K 1  ?  h  B    q  5  }    ?  w  a    ?    ^  ?  &  +  8  l  q  R    &  ?  f  5  ?  h    ?  q  ?  W  }  ?  ?    )  \  W  ?  ?  ?  |  ?  A  ?  H  ?  ?  o  ?  \  ?    h  ?  DI  DI  DI  DI  DI  DI  DI  DI  DI  DI  DI  DI  DI  DI  DI  DI  DI  DI  DI  DI  DI  DI  DI  DI  DI  DI  DI  DI  DI  DI  DI  DI  DI  DI  DI  DI  DI  DI  DI  DI  DI  DI  DI  DI  DI  DI  DI  DI  DI  DI  DI  DI  DI  DI  DI  DI  DI  DI  ?  ?  
          ?  ?  ?  ?    [  5    ?  ?  ?  Y  S  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  
               ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?              ?  ?  ?  ?  ?  m  U  @  /       ?  ?  ?      ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  }  i  U  A  V  X  U  N  @  )    ?  ?  ?  ?  v  S  .    ?  ?  ?  ?  h  ?  ?  ?  ?    !  *  1  2  +      ?  ?  ?  I    ?  l    ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  c  9    ?  ?  l  1  ?  ?  D  B  ;  0  #      '  /  .  "    ?  ?  ?  p  2  ?  ?  _  ?  ?  ?  ?  ?  ?  |  n  a  T  E  4  $      ?  ?  ?  ?  ?  )  ,  .  /  .  -  +  &  !        ?  ?  ?  ?  ?  ?  ?  ?  ?  Q  ?  ?  ?  ?  ?  ?  ?  Q    ?  U  ?  b  ?  i  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  r  q  }  ?  ?  y  D    ?  1  z  ?  ?    D  i  ?  ?  ?  ?  ?  ?  M  ?  ?  /  ?  K  ?    d  6  ?  ?  ?  ?  ?      (  0  8  A  I  Q  T  K  C  :  2  *  !  ?  ?  ?  ?  ?  ?  ?  ?  ?  q  Z  C     ?  @  ?  ?  ?  b  <               ?  ?  ?  	          ?      P  ?  ?  3  6  6  4  /  '    	  ?  ?  ?  ?  `  4    ?  ?  ?  ?  ?  ?  ?  ?    ?  ?  ?  ?  v  N  %  ?  ?  ?  s  D    ?  ?  ~  !  /  <  F  M  P  O  I  =  .    ?  ?  ?  ?  k  A    ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  {  a  H  (     ?   ?   ?  U  S  P  N  K  E  5  %      ?  ?  ?  ?  w  U  3     ?   ?  ?  ?  ?  ?            ?  ?  ?  ?  ?  ?  ?  J  ?  V  ?      /  J  R  Y  [  Z  T  E  -    ?  ?  ?  k    ?  ?    ?  j  =    ?  ?  ?  l  ]  W  d  S  1    ?  ?  u  D     ?  ?  ?    :  Y  n  t  s  }  ?  w  \  &  ?  ?  I  ?  t  ?  ?  (  <  >  9  .    
  ?  ?  ?  ?  ?  \  2    ?  ?  H    ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  u  c  O  9  #    ?  ?  ?  ?  ?  )  4  ]  q  {  ?  ?  {  g  O  3    ?  ?  D  ?  :  ?  ?  ?      !  "           ?  ?  ?  ?  ?  u  >  ?  s  ?  k   ?  ?  ?  ?  ?  ?  ?  ?  p  Z  @  &    ?  ?  ?  ?  {  \  =    ?  q  Z  A  #  
  ?  ?  ?  p  <    ?  ?  K     ?  ?  t  t  ?  ?  ?  ?  ?  ?  ?  ?  p  W  <    ?  ?  ?  m  <    ?  ~    d  Z  :    ?  ?  ?  J    ?  ?  d     ?  A  ?  ?  4  ?  ]  ?  o  ?  ?  0  ?    $    ?  ?    g  ?  q    
=  ?  ?    C  _  ?  ?    ,  K  _  f  R  A  c  0  ?  ?  X  ?  ?    ?  ?  ?  ?  ?  ?  ?  ?  l  O  3  G  <    ?  ?  \  ?  W  ?  ?  ?  ?  ?  ?  ?      ?  ?  ?  ?  ?  ?  o  =  ?  ?  1   ?  ?  ?  ?  ?  ?  ?  ?    2  7    ?  ?  ?  a    ?  ,  ?  ?  
X  
?  )  V  u  ?  ?  ?  c  =    
?  
I  	?  	  S  a    ?   '  ?  ?  ?  ?  ?  k  P  5    ?  ?  ?  ?  ?  ?  ?  j  T  >  (  ?  ?  ?  ?  ?  m  T  ;    ?  ?  ?  ?  ?    g  N  K  k  ?      ?  ?  ?  ?  ?  ?  ?  ?  ?  n  [  E  /      ?  ?  ?  ?  ?  l  N  *    ?  ?  z  A  ?  ?  Y    ?  ?  g  /  ?  ?  7  0  (        ?  ?  ?  ?  ?  j  A    ?  ?  ?  u  =    ?  ?  ?  ?  ?  {  d  L  3    ?  ?  ?  ?  s  8  ?  ?  e    |  r  k  e  `  X  Q  J  D  =  6  1  E  >  +    ?  F  ?  _  ?  ?  ?  ?  w  ]  ?  *    ?  ?  ?  n  8  ?  ?  v  K  '    ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ~  ?  ?      ?  ?  
f  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  }  j  V  C  1    
  ?  ?  ?  ?  ?  ?  i  7    ?  ?  t  V  5    ?  ?  ?  <  	  ?  ?  /  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ~  n  ]  ?  ?  ~  `  @    ?  ?  ?  z  J    ?  ?  ?  ?  a  D  0    ?  ?  ?  ?  k  R  7    ?  ?  ?  ?  {  T  ,    ?  ?  ?  ]  ?  ?  ?  ?  ?  ?  ?  E  
?  
?  
.  	?  	/  ?  ?  2  ?    ?  V  ?  ?  u  F    
?  
?  
8  	?  	?  	@  ?  ?  0  ?  m  
  ?  &  f  ?  }  L    ?  ?  ?  ?  ?  k  8    ?  ~  .  ?  ?  ?  ?   ?  4    
?  
?  
p  
=  
  	?  	?  	[  	  ?  q  ?  a  ?  %  G  {  ?