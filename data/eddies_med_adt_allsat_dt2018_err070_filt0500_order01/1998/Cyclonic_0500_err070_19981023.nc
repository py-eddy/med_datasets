CDF       
      obs    C   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?���`A�7       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       Mԉ�   max       Pj�4       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��S�   max       <�9X       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?&ffffg   max       @E�=p��
     
x   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ����
=p    max       @v���
=p     
x  +H   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @+         max       @R            �  5�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�R        max       @��@           6H   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �N�   max       <e`B       7T   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�mG   max       B-WJ       8`   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�~   max       B-?�       9l   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >)��   max       C��/       :x   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >A�t   max       C��        ;�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �       <�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          3       =�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          3       >�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       Mԉ�   max       Pj�4       ?�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?����l�D   max       ?�=�b��       @�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �$�   max       <�1       A�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?&ffffg   max       @E�=p��
     
x  B�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���Q�     max       @v�z�G�     
x  MP   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @+         max       @R            �  W�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�9        max       @��@           XP   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         E�   max         E�       Y\   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��ߤ?�   max       ?�=�b��     �  Zh               
               &      !   !      j      	      E            "   ,            K      
         )            "               7         
   %            �                     ,   ;         
                  O�RN2)�O&`N�:�N�w�M�6N��OɸNvX�O��Ny��O�iSO�@�O6FPV�DN��YO
j�N18Pj�4O��sO�I�O.A�OЦKOŶOAwOTN��gP:q�O+g�N��cNƴ@NV%O���O-w�O��sOo�O�{=O���N�uOC@-N�&�O���O���O��N�JO��O�߽N��OP�rP%7M�=�No0N�Y�O#ZgN�V�O���Oq�uO���N���O���Nh~N���NKE$N/�M��N��PMԉ�<�9X��o��o�ě��ě���`B�o�o�t��#�
�49X�e`B�u�u��o��C���C���1��1��9X�ě�����������/��/��/��/��h�����+�C��C���P���#�
�#�
�#�
�,1�,1�0 Ž<j�<j�@��@��H�9�L�ͽT���Y��aG��aG��u�u�u�}󶽃o��o��C���O߽�\)���-���w������{��������S�?BN[afgnnsg[XNMJDCB?imoz����{zmkiiiiiiiiKNW[gtuxvttg\[YNMKKK����������������������������������������������������������������������������������������������+/0;DEHJH@;7/-++++++H\fe~wn^UH</#����������������������������������������'/<HUdnrmlaU</#��������������������fdmz����������zma]^f��������������������otx�������������tojo����������������������#0Ibv���{bU<���amz�����������zke^\a���'&#����������R[ht��������tph`[UPRt�������������tnlmpt����/<E<0
�������HZacn������zna\UTLHH!)6;BEGDB1(!().1686)'�������)/:8/+
�����!#+/<HUafd`UP?<9/+"!dhqt����������trhhdd^amyzz�����zmda]^^^^�������������������������� ������������-8HUaenq{znjaUH@</(-6<=EIUan{�~znfaUH96��������������������5>Zu���������t[NB:55��������������������wz����zyuwwwwwwwwww�����������������������
"!
���������Y`gt���������tlgYTUYtx����������������ut���������������~{{�V[gty}���tg[QNVVVVVV�����������������������
$))���������������������������\g����������xtqtqg\\
)ANi�����tg[5
aafnvz���|zqnaaaaaaaY[dhjttxtjh[WSYYYYYY�������������������������������������������������������[dgt����������tqfVS[#&/<@>??BHNH</#� -/<EB<6422%
����FHIU\afeaa^UHEAAFFFFV]cft����������tg[RV?BDMOZZ[\[XOB@??????oz}���������}zoooooo��|vtqgffgjt|�������X[gliihgf[ZWXXXXXXXX!#01430-#"!!!!!!!!!!##',0;<<CFFFC=<0(###�������������������������������������ĿѿۿԿѿƿĿ������������������������������������������������߾M�H�A�=�A�K�M�Z�f�r�s�|�����v�s�f�Z�M�4�3�)�3�4�A�I�M�N�Q�R�M�A�8�4�4�4�4�4�4���������������������ĽнѽнͽĽ�����������������������������������������������������(�5�9�5�(�"����������z�u�n�h�b�i�n�zÇÓÙÝàåàÚÓÇ�z�z���������������������������������������������#�<�H�U�\�a�q�w�x�m�\�H�<�#��
������������������������������������������m�`�T�K�D�@�>�G�T�`�m�y�{�����������z�m�m�T�C�9�7�7�;�A�H�a�z��������������z�m��x�r�f�e�[�\�f�q�r��������������������ùéñ���������)�B�K�Y�]�^�b�U�6����I�F�I�V�b�b�o�{ǈǔǖǜǔǈ�{�o�b�V�I�I¿µ²¨¦¢¢¦²¿������������������¿�	�� �����������	������	�	�	�	�	�	�������x�i�a�[�a�s�����������������������	�����/�8�T�a�h�g�e�_�\�[�T�H�/�"�	ŠśţųŹ������������������ųŦŠ���
�����"�/�;�?�K�K�H�=�;�9�/�"���������������ʾ׾����������׾���ùìàÛÖÚàìò��������������������ùøìéì÷��������������������������ùø�������s�n�l�s���������������������������������������������	�������	�����俒���z�t�t�z�����ʿѿݿ������ѿĿ����6�2�)�(�%�!���)�6�B�I�O�[�]�[�S�O�B�6�/�*�#�!��#�/�:�<�@�H�U�U�U�L�H�<�<�/�/������������ �����(�$�����������������������	���������������������������*�6�C�L�T�Z�_�\�O�C�6�*��M�A�9�9�>�A�L�M�Z�b�c�f�m�n�l�f�c�^�Z�M�g�^�Z�N�A�)�%�-�N�Z�g�s�������}�|�x�s�g�I�<�0�#� �#�.�0�<�@�I�U�b�n�w�s�n�b�U�I�=�0�������������$�0�;�I�T�e�f�`�V�I�=�������h�d�i�l�x���������ûллɻû������������������������������������������������������#�/�<�?�M�R�Q�H�<�/�#��
�������������������������������������������A�5�,�%�%�3�A�N�g���������������s�g�N�A����������������������������������������Ƴƫƽ���������$�0�2�&�������������z�u�n�u�z�������������������z�z�z�z�z�z�L�I�K�@�?�A�L�Y�e���������������~�r�e�L��v�h�f�r���������ʼּۼ˼����������������ĿĶĳİİĳĿ����������������������������������������,�5�E�N�X�N�B�5�,����[�O�I�K�[�a�h�tāčĚĭĺĿĺĵĦč�t�[�ѿοĿ������������Ŀſѿӿҿѿѿѿѿѿѹ��������������ùǹϹѹӹϹù������������r�q�p�r�z���������������������r�r�r�r�ʼ��üʼԼ���������������ּ��Z�Z�\�b�g�i�r�s���������������|�s�q�g�Z¼¦¦¿����������������¼E�E�E�E�E�E�E�E�E�E�E�E�FF
FFE�E�E�E�ED�D�D�D�D�D�D�D�EEEE*E7E?E8E*EEE��ؾ׾о׾ݾ������	���	���������z�n�a�H�2���%�/�H�a�zÊÄÍÑÖÓÇ�z����������������������������������������F1F*F$F"F$F$F1F=FJFTFOFJF=F4F1F1F1F1F1F1ƎƏƚƧƩƧƝƚƎƅƁ�zƁƊƎƎƎƎƎƎ�#���#�0�<�I�L�I�A�<�0�#�#�#�#�#�#�#�#�����������ĽнĽ�������������������������	����߽ݽ۽ݽ������(�*�2�(����������������������������������� a R  N / � p  9 B x 2 F I J x ; v E h b ; : N W A | S F 0   + . Q ; T g 6 W D W ; d T = 3 E l N M � 4 L > � K S X : R D Q = f g P y    G  V  9  �  �  d  k  2  w  K  �    �  R  �  2  0  �    %  g  y  �  �  �  �    �  �  �  �  l  X  �  #  s  �  &  I  �  �  �    �    �  n  �  �  )  �  m  �  s  [  �    �  �  �  j  �  Y  m  "  6  F<e`B��o�u�e`B�u�#�
�#�
�����49X�D����t��@��D���0 Ž��ٽC�����������Q�H�9�aG���P�u��\)�<j�0 Žo�����q���''@������}�T���T�������\)�<j��o�y�#�����t���\)�e`B�� Ž��-�m�h����N��ixս�t���hs���㽗�P��vɽ�/�%����ȴ9��-�ȴ9��^5��^5�Ƨ��xս�x�B�OA�j/B��B	�B ��B��B�B�A�mGB�B[�B��B�B 
B ||BFBZB!��B&:�B }�B��B,�B'B��BϴB+�B�B��B�B{�A���B_9B'�B~NB#�B�B	P B ��B (B!w4B�~B	ؑB �B�B	XLB"$B�kBĪB
�GBf�B�uB��B,X�B-WJB��B
B�B[�B�TB
�}B�?B�*B	��B	
B%�WB& B�tB�tA�m�B��B�xB ��B;!BSB6�A�~B=�B��B�B  B�*B }�B;�B�XB!�lB&Y�B ��B=�B�4B��B�B+B�#B�(B��B�VBBA�
B��B@^B�UB��BLB	��B ��A���B!D�B=�B	ũB �B��B	�B"@0B�B�=B
~�B?�BB�8B,>�B-?�B��B
?sB��BA)B�CB
�oB��B?�B	��B��B%EB%��B�Av��A�uWA?�DA:�A#��A�!�A��A��A��A��A���AjX
A��@��A��vB�
A��yA�JA���A��A�A��7AQMA�etA�ZGA���A��Ax)oA׶A�B�A�[B��A�B�A=��A�aA���B
�@��'A��A���A�\A�84A���BʹA�Č?�@�]A�A�<�A�D|Ay�>)��@�ЏA�pA���A���C�j�C�Y�AWCA�؈@�l!C��/B�A�[yA$�A1܀@U��Ax��A�~�A@��A:��A%�A�~�A��'AȄaA�ݾA��A�tAi0�A���@���A���B>	A���A�e�A��^A� A�pA�~�AP�A΂�Aρ�A�m_A��Au�A�r$AA�B@A��EA=*A�j.A�B	�q@��(A�x�AÂ�A䃎A���A�
BD A�{?��>@�>*A�~�A�{�A݀Ay0|>A�t@��ASnA��_A�|�C�i>C�H�AX�)A�U@��-C�� B*&A�h�A#-A3f@U �   	         	   
               '      "   "      j      
      F             #   ,            L         	      *            "               8         
   &            �                     -   ;                                                         '         !      3            3   #   '      %               1                           )                  %   #         !      /   +                              #                                                   !               %            3   !   !                     '                           #                     #               !                                                      N��N2)�O&`N�:�N���M�6N��N�Q�NvX�O��'Ny��OO�$�O6FO��N�6�O
j�N18Pj�4O�\�O���O.A�O!r�O��UO�	OTN��gP�O+g�N��cNƴ@N3�UOF�N��O|��Oo�O�h:OuWFN�uOC@-NT"�OZ�wO$�gO��N�JO���O�,kNYq�O��O6wqM�=�NR}�N�Y�O#ZgN�V�O�z�O��OD[dN���N�DNh~N���NKE$NJM��N�W!Mԉ�  +  �  �  �  �  m  �  �  Y  �  y  �  �  �  ?  r  *  s  �  �  �  �  �  ;  �  �  �  	w  �  D  ]  y  �    �  �  +  	      6  	  �  g    4  `  �  Z  �  g  �  c     �  M  
*  
w  �  �  �  �  B  �  =    <�1��o��o�ě���`B��`B�o�T���t���t��49X���ͼ�9X�u�D����t���C���1��1���ͼ��������'+����/��/�49X�����+�\)�0 Ž,1��w�#�
�0 Ž',1�,1�<j�q���]/�@��@��L�ͽT���Y��m�h�$ݽaG��y�#�u�u�}󶽇+������
��O߽��T���-���������� Ž���\��S�FN[`dgmmqg[YNNKGFFFFimoz����{zmkiiiiiiiiKNW[gtuxvttg\[YNMKKK��������������������������������������������������������������������������������������	��������+/0;DEHJH@;7/-++++++#/HU_b_aqnaH</#����������������������������������������"(+0<HUgkdaXUH</&#!"��������������������qwz�����������mjggjq��������������������otx�������������tojo����������������������#0Ibv���{bU<���_bjmz���������zmgea_����##!���������R[ht��������tph`[UPRtv{�������������utst����#/83)	�������PUamz������znia_WUQP!)6;BEGDB1(!().1686)'������!#"
��������!#+/<HUafd`UP?<9/+"!dhqt����������trhhdd^amyzz�����zmda]^^^^����������������������������������������3<=HUaakmaaUIH<133337<>FJUanz~|xniaUH<7��������������������AN[gt�������t[NB=88A��������������������wz����zyuwwwwwwwwww����������������������
 
���������[_gt����������tgc[[[{����������������y{���������������~{{�V[gty}���tg[QNVVVVVV������������������������	#') ��������������������������`gt������������tg`^`)5BR[^[ZNKB5)aafnvz���|zqnaaaaaaaY[hrtwtih[WTYYYYYYYY�������������������������������������������������������W[`fit���������tg`YW#-/3;;<==<;/#�#/344/,%#
���FHIU\afeaa^UHEAAFFFF~�������������zyxy~~?BDMOZZ[\[XOB@??????����������~z����������|vtqgffgjt|�������X[gjhhg[[XXXXXXXXXXX!#01430-#"!!!!!!!!!!##%'-0<AEFEB=<0(####�����������������������������������ĿѿٿӿѿſĿ��������������������������������������������������߾M�H�A�=�A�K�M�Z�f�r�s�|�����v�s�f�Z�M�4�3�)�3�4�A�I�M�N�Q�R�M�A�8�4�4�4�4�4�4���������������������ĽϽɽĽ���������������������������������������������������������(�5�9�5�(�"���������Ç��z�n�k�f�n�v�zÇÓÕØØÓÉÇÇÇÇ�����������������������������������������
�����#�/�<�A�H�U�X�d�d�_�S�H�<�#�
����������������������������������������`�]�T�R�K�M�T�`�m�w�y���������}�y�m�`�`�m�a�T�H�=�;�;�H�T�m�t�z�z�����������z�m��x�r�f�e�[�\�f�q�r��������������������������������)�9�B�M�P�O�H�B�6�����V�L�V�b�e�o�{ǈǔǕǛǔǈ�{�o�b�V�V�V�V¿µ²¨¦¢¢¦²¿������������������¿�	�� �����������	������	�	�	�	�	�	�������x�i�a�[�a�s�����������������������"��	�����1�:�T�a�a�\�Z�X�Q�H�;�/�"ŭŤŤťŴŹ������������������Źŭ���
�����"�/�;�?�K�K�H�=�;�9�/�"��ʾ������������������ʾԾ����������ùìãßÛÜàìù��������������������ùùõíù������������������������������ù�������s�n�l�s���������������������������������������������	�������	�����係�}�}���������Ŀݿ�������ѿĿ��������6�2�)�(�%�!���)�6�B�I�O�[�]�[�S�O�B�6�/�*�#�!��#�/�:�<�@�H�U�U�U�L�H�<�<�/�/������������ �����(�$������������������������������������ �����*�6�C�G�N�O�O�C�0�*���M�C�A�;�;�A�A�M�Z�f�f�i�f�f�_�Z�M�M�M�M�g�`�Z�N�A�+�'�/�N�Z�g�s�������{�z�w�s�g�I�<�0�#� �#�.�0�<�@�I�U�b�n�w�s�n�b�U�I�0�$�����������0�8�F�Q�c�c�\�V�I�=�0���������|�x�i�d�j�l�x���������ûλĻ����������������������������������������������������#�/�<�?�M�R�Q�H�<�/�#��
�������������������������������������������N�A�3�,�/�5�=�A�N�Z�g�p�s�����y�s�g�Z�N������������������������� ����������������Ƴƫƽ���������$�0�2�&�������������z�u�n�u�z�������������������z�z�z�z�z�z�Y�L�A�?�B�L�Y�e�����������������~�r�e�Y��y�r�j�i�r������������ż������������ĿĸĳııĳĿ����������ĿĿĿĿĿĿĿĿ�����������������������&�4�5�.�!�����h�d�a�f�l�t�~āĊčęĚĝĞęčċā�t�h�ѿοĿ������������Ŀſѿӿҿѿѿѿѿѿѹ����������ùƹϹйҹϹù����������������r�q�p�r�z���������������������r�r�r�r�ʼ��üʼԼ���������������ּ��Z�Z�\�b�g�i�r�s���������������|�s�q�g�Z����²¦¦¿����������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�FFE�E�E�E�ED�D�D�D�D�D�D�D�EEEE!E*E6E2E*EEE��ؾ׾о׾ݾ������	���	���������/�+�&�/�<�@�H�O�U�a�n�r�n�n�a�U�H�<�/�/����������������������������������������F$F#F$F%F1F=FJFSFNFJF=F1F$F$F$F$F$F$F$F$ƎƏƚƧƩƧƝƚƎƅƁ�zƁƊƎƎƎƎƎƎ�#���#�0�<�@�=�<�0�#�#�#�#�#�#�#�#�#�#�����������ĽнĽ�������������������������������߽������(�)�1�(������������������������������������ X R  N 1 � p & 9 0 x . 7 I : s ; v E \ M ; C I U A | V F 0   6 $ G 5 T a 8 W D 4   D T = * : I J A � : L > � I E K : V D 3 = W g P y      V  9  �  �  d  k  �  w  x  �  3  �  R  K    0  �    �  �  y  x  �  h  �    �  �  �  �  X  �    �  s  �  �  I  �  q  �  i  �    N    �  �  �  �  V  �  s  [  t  k  �  �    j  �  Y  !  "    F  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  $  '  *  &        �  �  �  �  �  �  �  �  �  �  �  �    �  �  �  �  �  �  |  w  u  s  q  o  l  _  <    �  �  �  �  �  �  �  �  �  �  �  �  �  �    o  _  M  8    �  �  �  �  �            �  �  �  �  �  �  �  �  |  t  o  m  q  u  �  �  �  �  �  �  �  �  �  o  L  #  �  �  �  �  _  2   �   �  m  ^  P  B  3  %      �  �  �  �  �  ~  k  W  D  1    
  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  h  C    �  �  Z    �  �  y  N  #  Y  S  N  I  D  ?  :  5  /  *       �   �   �   �   �   �   �   �  e  }  �  �  �  �  �  �  �  �  ^  "  �  {    �  #  �    <  y  d  O  ;  (      �  �  �  �  �  �  �  �  �  �  �       �  �  �  �  �  �  �  �  �  �  �  �  Z  %  �  q  �  n  �  �  =  a  �  �  �  �  �  �  i  I  $  �  �  �  r  )  �    �  �  �  �  �  �  �  �  �  g  8    �  ~  5  �  �  �  �  D  �  �  
�  7  �  �    9  >  9  .  #  �  �  ,  
}  	�  �    �  \  k  Q  n  o  e  Q  2    �  �  U    �  x  *  �  �  )  �  d    *  %  !          	  �  �  �  �  �  �  �  �  �  �  n  V  s  i  `  V  K  9  (      �  �  �  �  �  k  S    �  �  8  �  �  �  �  b  /  �  �  �  E    �  �  s  K    �  0  [  >  {  �  �  �  �  �  �    j  M     �  �  ^    �  ^  �  �  �  u  �  �  �  y  \  A  '    �  �  m  (  �  b  �  �  �  �     �  �  �  �  �  �  �  �  w  ^  D  &    �  �  w  C     �   �  �  /  d  �  �  �  �  �  �  �  �  �  w  6  �  �  >  �  �  W     1  :  :  5  !  �  �  �  _    �  m  *  �  �  7  �  �  O  �  �  �  �  �  �  �  �  �  x  Q     �  �  �  D  �  u    �  �  �  y  n  `  S  H  K  >  &    �  �  �  n  K  /    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  	  	J  	c  	q  	v  	j  	R  	/  	  �  �  g  H    �    �  �  �  )  �  �  �  �  �  �  �  �  �  h  8    �  �  D  �  �  =  |    D  8  +      �  �  �  �  �  �  �  �  p  \  H  A  =  $  	  ]  S  H  =  1  $      �  �  �  �  �  n  R  5    �  �  �  u  x  �  �  �  �    7  v  �  �  �  �  �  �  �  �  �  �  �  8  [  �  �  �  �  �  �  T  $  �  �  z  J    �  !  �    V  �  �  �        �  �  �  �  �  �  d  ,  �  �  2  �  @  5  �  �  �  �  �  �  r  `  L  <  (    �  �  �  m  @    �  �  �  �  �  �  �  �  u  [  E  F  D  A  8  '    �  �  �  %  �    )  )      �  �  �  �  �  =  �  �  C  �  �    �  �         �  �  �  �    �  �  �  n  :    �  �  k  )  �  �   �          	    �  �  �  �  �  �  �  �  �  �  �  �  �  �    �  �  �  �  �  �  s  �  �  d  D  .        �  J  �    �    "  3  )    �  �  �  {  Q  %  �  �  �  G    �  v  C  b  �  �  	  	  	  	  �  �  �  f  $  �  �  "  �    i  5  v  �  �  l  N  +  2  �  �  �  �  �  x  I    �  Q  �  �  S  �  g  f  a  H  $  �  �  �  �  d  #  �  �  1  �  �  d  '  �  ]    �  �  �  �  �  �  w  e  U  G  :  ,        ,  H  4       1  *        �  �  �  �  t  D    �  �  �  L    �  n  E  E  T  I  8  %       �  �  �  �  i  4  �  �  l  S  -  �  �  �  �  �  �  �  �    d  I  *    �  �  �  n  B  	   �   �    K  K  N  X  T  G  5    �  �  �  a    �  �  ,  �  r  X  -    �  s  �  4  a  �  �  �  �  �  F  {  r  3  �  
o  �  �  g  a  Z  S  L  E  ?  8  1  *  /  @  R  c  t  �  �  �  �  �  �  �  �  �  �  �  �  �  �  h  @    �  �  a  &  �  �  \    c  V  H  :  $    �  �  �  }  U  +     �  �  h  1  �  �  q     �  �  �  �  �  �  t  X  :    �  �  �  �  =  �  �  =   �  �  �    \  8    �  �  �  �  �  �  g  1  �  �  �  �  M  �  :  M  I  @  4  $    	  �  �  �  �  q  B    �  �  w    �  	�  	�  	�  
  
(  
  	�  	�  	�  	>  �  _  �  s  �  O  �  �  �  W  	�  
  
n  
n  
v  
u  
`  
'  	�  	�  	�  	^  	  �  Z  �  �  (  ~  �  �  �  �  �  q  ^  J  5      �  �  �  �  W  �  �  �  o   �  ]  G  +    �  �  �  �  �  �  �  �  �  �  9  �  `  �  N  �  �  �  �  �  �  �  �  �  �  u  O  ,    �  �  �  K    �  �  ;  �  �  �  �  |  j  U  :    �  �  �  2  �  s    �  7  �  B    �  �  �  Y    �  �  S    �  �  a  (  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  =  ;  :  8  7  5  4  /  )  #               �  �  �  �        �  �  �  �  l  E    �  �  �  k  1  �  �  L  �  o    	  �  �  �  �  �  �  �  �  |  l  \  L  :  (      �  �