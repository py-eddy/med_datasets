CDF       
      obs    L   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?��1&�y     0  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��   max       Pp�     0  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��1   max       <�     0      effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?J=p��
   max       @E�p��
>     �  !<   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��
=p��    max       @vrfffff     �  -   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @/         max       @Q�           �  8�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @ɯ        max       @�          0  9�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��   max       <�`B     0  :�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B+     0  ;�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�|   max       B+��     0  =$   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >ș�   max       C�Rx     0  >T   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >���   max       C�Vz     0  ?�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          8     0  @�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          =     0  A�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          9     0  C   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��   max       Pn��     0  DD   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�$tS��N   max       ?�Xy=�c     0  Et   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��1   max       <�     0  F�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?J=p��
   max       @E�p��
>     �  G�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�G�     max       @vo33334     �  S�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @&         max       @Q�           �  _�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @ɯ        max       @��         0  `,   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         C�   max         C�     0  a\   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��X�e   max       ?�Xy=�c     �  b�                                       "                            1               
      
         "          8            '   !            	                     	                            &            7      (   
                     #   %   N	��M�oZN�VN��NAZO!�sN�8O?%�N'��N�7N�=O�3�PFǪO]ݹM�%�O��NiпO�K�N8�tPN�SO�חPn��O;O��N�e�O	O��O�10NʞXO��CN��VOK^\O1�5N*� Pp�O��XNDm�N���Ph�O�:�N�N�N�WO-�<N>��O\�Oa�N�
DO���O��M��N���N�L-O�N��XO��N��N'��O�D8O���O�&rOlO���O-�yOͅ�O�naO�!0O$�O�գN�%�N��N/�tN>U�N��O�	O�U0N��<�<�h<T��<T��<49X<t�;��
;D���o��`B��`B�o�t��T���e`B��o��t���t���1��1��1��j�ě��ě��ě��ě����ͼ��ͼ��ͼ��ͼ��ͼ�����/��`B��`B��h��h��h���������������o�o�+�C��\)����w�''''0 Ž0 Ž49X�8Q�D���P�`�aG��e`B�ixսm�h�q���q���y�#�}󶽁%��o��o��+���罬1���������������HHINUVUSHCCFHHHHHHHHqt���������}utqqqqqq@IUUbnonmhbYUIH>@@@@

BKN[gkqpolhg\[NKECBBGHHUannz{}znaULHGGGG&).5?BNSZZUZNB5)'!"&yz{������zyyyyyyyyyyfmoz��������zsmmffff������������������������������������q��������������tonlq������������//76<HIHEF</////////���������������������������������������������)�������;;DHKIHG=;;:99;;;;;;mu���������������zmm��������������������#In{������nb<		�������������������� )5BN[t�������tgNB5 ��������������������_acnz��������znmfa`_RT_acmtz��ztma]TTRR���������������������������������������������	�������Z[gt~�{tg[WWZZZZZZZZ!6>BOX[`b_[OB60)"#)/<DFHIJHC4/#""����������������������������#`ZI0����������������������������������������������������������������et���������������tdedt|�������������ta]d��������������������)59ABDEDB;5/)46:BOU[bhmmja[OB6134DO[hjqh[OHDDDDDDDDDDmtv�������������tmmm��������������������������������������������"'('#
�������@[htx����th_[OLB=98@������������������������������������������
#/*# 
�������W[t����������tgYZVUW�����������������������
#0:ACCB<;0
���BIJNUV[`bcb\UIC@>==B��������������������@EOV[ho{}ywth[XOGB>@+/<UZgidZcUHC90.('(+�����586)�������45BNZ[_df^[NB=:87544lt������������tnmjgl>BN[`gisrhg[PNNBB::>����������������������������������������)BN[k\PC?1)
#0<GHC<0#

	Sbn����������{mbUKMS��������������������������������������������������������������������������������uz�����zwvuuuuuuuuuu���$)'������6:@ELPOB6))168<FHKNROH<9787888888¿¶²±²¿��������¿¿¿¿¿¿¿¿¿¿ĿĿĿ������������������ĿĿĿĿĿĿĿĿ�����������!������������������������������������ʼѼּؼ׼ּʼ������������������������������������������������������z�|�����������ĿſĿĿ������������M�A�A�7�7�@�A�M�M�Z�b�f�f�d�Z�R�M�M�M�M�N�H�A�:�5�3�0�5�A�N�Z�g�s�w�x�t�s�g�Z�N���������������������������������������������������������� � ������������ŹŷŭŠŠŪŭŹ����������������������Ź�$��������������0�=�H�N�L�F�=�5�0�$�s�S�J�P�Z�s���������������������������s�n�j�l�q�v�{ŁŇŔŭůŵŭŭŧŢŠŔ�{�n�<�<�/�#��#�#�$�/�<�?�<�<�<�<�<�<�<�<�<������������z��������������¾ľƾǾ����n�l�c�a�]�a�n�w�zÇÊÇÀ�z�n�n�n�n�n�nÓ�z�n�]�Y�[�c�q�zÀÇÓÙðÿ��üùàÓ�������������������������������������������������������ѿ����A�T�Z�T�D�?�(��	�������������	��"�2�D�S�T�Y�P�.��	���{�k�f�i�l�t���������������������������a�U�H�#���
�������
��#�/�B�H�U�k�f�a�������������	��"�'�/�.�*�����z�o�m�j�m�r�z�������������������z�z�z�z���|�z�o�s�z�����������������������������	���������	���"�/�8�;�>�6�/�#�"��	�;�"���	� ��	��!�;�H�T�[�c�c�\�R�H�;���������������������������������������˿���׾˾;׾���	���.�8�;�=�5�.���*�(�"�&�*�6�C�G�M�J�C�6�*�*�*�*�*�*�*�*�r�o�f�I�@�=�:�@�E�M�Y�f���������z�t�rE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�ŭūūŭŶŹ��������ŹŸŭŭŭŭŭŭŭŭ�_�F�-���ʺǺͺ��-�F�b�]�e�x�����x�_�h�O�@�?�B�O�[�hāčĚĝĤĦĚĖčā�t�h�N�F�B�8�7�B�G�N�T�[�f�[�N�N�N�N�N�N�N�N�T�J�G�E�G�K�T�[�`�k�m�u�y�����y�m�`�T�T��ľĭĨĳĿ���������
��#���4�&������{�o�b�I�?�7�6�<�I�b�{ŇŔŠŬŵůŔŇ�{�O�C�<�6�+�*�����*�6�C�C�O�O�O�O�O�O�����������������������������������������"��������������	��"�.�3�;�G�G�;�.�"�����������ûλλǻû������������������������������{�}��������������������������Ɓ�u�h�a�a�[�d�d�h�s�uƁƔƚơƟƚƘƎƁ�O�C�C�8�6�4�6�C�O�T�\�d�a�\�O�O�O�O�O�O�������������Ŀѿݿ������ݿ˿Ŀ������������������˽�������������ݽнĽ�������������������������r�q�r�t�r�r�r�������������r�r�r�r�r�r����������������� ������������������������������#�5�G�R�a�f�B�5�)���Y�S�L�G�A�@�>�<�@�L�L�Y�\�e�m�q�m�e�Y�Y�н������������������нݽ�������ݽн���������(�/�4�7�A�E�A�4�(�����D�D�D�D�D�D�D�D�ED�D�D�D�D�D�D�D�D�D�D߹Ϲù����������ùϹܹ�������������������(�Z�s���������������Z�N�5�(��ĿĳĚā�z�s�z�~�vāčħĴĺĿ��������Ŀ�6�,�/�,�6�6�B�O�[�h�j�o�n�h�[�O�B�=�6�6²¿������������¿²�#����#�.�/�<�H�U�V�]�a�b�a�U�T�<�/�#�m�_�U�S�l�x�����������ûл�ػû������m���~�t�t�}���������ֺ����!����ɺ�������ùíâàÚçì����������������һ������������%�%�������������������4�A�M�R�O�F�4�#������������������������
�����������������<�:�<�G�H�U�a�f�a�[�U�H�<�<�<�<�<�<�<�<�����������������ĽнٽнĽ�������������àØÓÉÓàìñôìàààààààààà��ݹ������������������������ڼ�����!�.�:�H�\�_�S�G�:�.��Z�g�s�������������������s�g�N�A�6�8�A�ZD�D|D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D� \ { L Q B . A ) I ; [ N ) ? \ F N Z z R  , � ^ . E 9 5 M I / @ > K V 9 � ( " I @ S K x ? . F c 8 � L A J b 8 e p 0 q � 4 & & @ s S M J D c { V R N l h    W  �  �  O  f    �  [    >  c  D  �  D  J  y  1  �  �  y  �    �  �  4  >  A  �  i  �  �  �  X      d    �    �  &  �  �  `  �  �  �  W  /  �  �    
  _    �    �  "  U  W  t  �  Q    r  �  �  Z  �  ?  .  �  O  �<�`B<���<o;ě�;�o�ě��t��T���ě��e`B�T���o�0 Žt���o��/�C��@���j�Y��H�9��hs��w�L�ͼ����\)�]/�\)�H�9���q���m�h������q���o�t���O߽�%���0 Ž0 Ž#�
�D���#�
��P�Y��u��P�@��q���}�@��}�aG��m�h�������P���C���7L��hs��S����T�Ƨ�C�������hs��+��7L������7L���ͽ��Ƨ�B�B=�B
R�B'P�B$�aB�HB9B��BѕA�� B�BB�B�RB��B!�	BO�B^#A���BWIB�&B'	<B�}B	eB�?B2A�&�BO�B��B�VB	\�BI~BX&B'
B$&�B��B:�B+B
��B
�gB��B�B��B0BABx�B;BW�B��B!�B1�B�B
M'B!8�B$��B&�+B�Ba(B�B�'BeB
�1Bw�B7B�HB"�B%a�B(ȜBE�B
�.B��BzBn�B�BySB�SB93B>�B
?�B'�B$�B�EB=Br�B�B !�B�nB=pB��B��BɦB!�B��B�6A�|B'IB�[B&��B�.B	�B¢Bj�A�w7BC�B�4B�B	9�BO�B@RBOVB#�dB��BL�B+��B
"B
��B�$B�B�/B�wB7B?�B@B?ZB�xB!��B�=B?�B
ēB! oB$�}B&��B?�B@B�mB�BBkB
6�BB�B>�B��B�oB%��B)?�B>�B
��B9�B;jBy�B��BB.B��A��WA�'�A���@�y�@���As�A=��A�5fA��B��A��2B	�xA��?A��AQAJ��A��A���A�MgA�j%A^�'A�w�A��+A[�A�[�A��_A�!2A��?A���AZ��B I�@�	RC�RxA��G@tбA�@~A���AipA�J;A�-�B #ZA�N�A^˛@��@�%B$�B2#Ax�rA*ٕ@�d?@�^A��A�ʻ?�A&�lA5@sC�9D>ș�A��A߸5A�`�A���Aë@��@0� A�߰@�"�@Ƣ�B��A���A%U�A˜�?.�A
=A��vC���A���A䙯A��~@�.�@���As)A=�A��=A�$B��A�o�B
@^A�|A�dA��yAKp�A��A˘/A��A�|�A^�PA�$A���A[�A�u�A�~WA���A�anA���A[�B �{@ڛ�C�VzA�ӳ@h�]A�	AA��Ai�=A�dA��A���A�(A_*b@�u@���Bx6B8�Ax��A*=@�n�@��A���A��?�*|A$�A5�C�D>���A��pA�|RAمkA�\�A��@��@�[A�MQ@��@��(B��AĀA%��Aˀ�?<��A	,A���C��~                                       #                     !      1                              "          8            (   "            
      	               
                     !      &            8      (   
                     #   &                                          -               #      5      3      !                                 =            -   %                                       #                  )   +            #   '   %      !                                                               -               !      '      3                                       9                                                      #                  #               #   '   #      !                        N	��M�oZN�VN��N�O��N�8O?%�N'��N�7N�=O;v�P8ZO]ݹM�%�O��N;�O�onN8�tPQ�OIC�Pn��N׊IO6]~N�e�O	N��O��NʞXO�i�N��VOtEN�ߠN*� P`�MO^�^NDm�N���OJ@�O���N�N�N�WO-�<N>��O\�Oa�NH��O���Ou7M��N���N�L-O�N��XO�`�N��N'��OG�O�X�O/�YO�\O���O-�yOͅ�O�naO�UhO$�O��N�%�N��N/�tN>U�N��O$&O�U0N��  M  k    �  `  �  e        �  Y  �  �  �  �  q  P  F  "  �  �  i  -  �    B  �  q  �  �  
  H  H  D  �     D  c  �    �  �  i  �  L  g  l  s  �  �  �  |  F  �  �  {  �  "    �  �  �    	�  �  
  �  �  *  t  \  �  �  
  �  �<�<�h<T��<T��<#�
<o;��
;D���o��`B��`B�T���49X�T���e`B��o���
�ě���1��`B����j��/�o�ě��ě������t����ͼ������ͽo�\)��`B���o��h��h�D����w�����������o�+�+���\)����w�''0 Ž'0 ŽT���H�9�ixսH�9�P�`�aG��e`B�ixսu�q���u�y�#�}󶽁%��o��o���w���罬1���������������HHINUVUSHCCFHHHHHHHHqt���������}utqqqqqq@IUUbnonmhbYUIH>@@@@

	LN[giponkg[NLFDCLLLLGHHUannz{}znaULHGGGG&).5?BNSZZUZNB5)'!"&yz{������zyyyyyyyyyyfmoz��������zsmmffff���������������������������� �������t��������������vqott������������//76<HIHEF</////////������������������������������������������������������;;DHKIHG=;;:99;;;;;;{���������������zus{��������������������#In{������nb<		��������������������JNY[gt�������tg[NLDJ��������������������_acnz��������znmfa`_TTamrz|��zomla^UTSTT����������������������������������������������������Z[gt~�{tg[WWZZZZZZZZ&).6BOS[]_[XPOB=6)&& #'///<<CED<0/.#  �����������������������������#^XI0����������������������������������������������������������������otw�������������ztno|��������������toji|��������������������)59ABDEDB;5/)46:BOU[bhmmja[OB6134DO[hjqh[OHDDDDDDDDDDmtv�������������tmmm��������������������������������������������"'('#
�������J[htv{������th[ODCBJ������������������������������������������
#/*# 
�������W[t����������tgYZVUW������������������������
#07>AA@<0
���BIJNUV[`bcb\UIC@>==B��������������������KOV[hhtutproh[SOJHFK,/<MUZ]egebVUH<62,*,������
�������55BNY[_cf][NB>;88555lt������������tnmjgl>BN[`gisrhg[PNNBB::>����������������������������������������)BNRaZOB=/)
#0<GHC<0#

	RUWfn{����������{nbR��������������������������������������������������������������������������������uz�����zwvuuuuuuuuuu���" �����6:@ELPOB6))168<FHKNROH<9787888888¿¶²±²¿��������¿¿¿¿¿¿¿¿¿¿ĿĿĿ������������������ĿĿĿĿĿĿĿĿ�����������!������������������������������������ʼѼּؼ׼ּʼ����������������������������������������������������|���������������ÿ������������������M�A�A�7�7�@�A�M�M�Z�b�f�f�d�Z�R�M�M�M�M�N�H�A�:�5�3�0�5�A�N�Z�g�s�w�x�t�s�g�Z�N���������������������������������������������������������� � ������������ŹŷŭŠŠŪŭŹ����������������������Ź�0�,�$�����������$�0�=�C�J�G�@�=�0�s�W�M�R�s�����������������������������s�n�j�l�q�v�{ŁŇŔŭůŵŭŭŧŢŠŔ�{�n�<�<�/�#��#�#�$�/�<�?�<�<�<�<�<�<�<�<�<������������z��������������¾ľƾǾ����z�p�n�e�a�`�a�n�v�zÇÈÇ�}�z�z�z�z�z�zÒÇ�n�b�]�a�d�n�yÇÓÝéùÿýùìàÒ������������������������������������������ѿʿſ¿����Ŀݿ�����A�G�O�2�(���	�������	���"�.�8�;�E�I�=�;�.�"��	���{�k�f�i�l�t�����������������������������
����
��#�$�/�<�>�H�S�H�<�/�#�����������������	��"�#�%�"����	���z�o�m�j�m�r�z�������������������z�z�z�z���|�z�o�s�z�����������������������������	�����	���"�/�5�;�<�;�4�/�"���	�	�/�'�"� � �"�/�;�H�T�Z�[�T�S�H�;�/�/�/�/���������������������������������������˿�����׾̾ξ׾���	���.�7�;�3�.���*�(�"�&�*�6�C�G�M�J�C�6�*�*�*�*�*�*�*�*�f�b�Y�M�@�=�@�J�M�Y�g�r������}�t�r�fE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�ŭūūŭŶŹ��������ŹŸŭŭŭŭŭŭŭŭ�l�_�F�-���ͺȺк��-�F�^�Z�c�����x�l�h�[�O�B�A�B�S�[�hāčĚğğĚđčā�t�h�N�F�B�8�7�B�G�N�T�[�f�[�N�N�N�N�N�N�N�N�T�J�G�E�G�K�T�[�`�k�m�u�y�����y�m�`�T�T���������������������������������������n�h�b�S�P�U�b�k�{ŇŔŠţŭŮŦŠŔŇ�n�O�C�<�6�+�*�����*�6�C�C�O�O�O�O�O�O�����������������������������������������"��������������	��"�.�3�;�G�G�;�.�"�����������ûλλǻû������������������������������{�}��������������������������Ɓ�u�h�a�a�[�d�d�h�s�uƁƔƚơƟƚƘƎƁ�O�J�C�:�>�C�O�Q�\�b�]�\�O�O�O�O�O�O�O�O�������������Ŀѿݿ������ݿ˿Ŀ������������������Ľнݽ����������ݽнĽ�������������������������r�q�r�t�r�r�r�������������r�r�r�r�r�r����������������� ������������������������������#�5�G�R�a�f�B�5�)���Y�S�L�G�A�@�>�<�@�L�L�Y�\�e�m�q�m�e�Y�Y�нĽ������������������Ľн۽ݽݽ��ݽн���������(�/�4�7�A�E�A�4�(�����D�D�D�D�D�D�D�D�ED�D�D�D�D�D�D�D�D�D�D߹ù����������ùϹѹܹ�����������ܹϹ������%�5�A�Z�g�����������~�g�N�5�(�Ěĕčāā�}�}āąčĚĦĭĳĴķĹĳĦĚ�6�3�0�-�6�8�B�O�[�h�j�n�m�h�[�O�B�7�6�6²¿������������¿²�#����#�.�/�<�H�U�V�]�a�b�a�U�T�<�/�#�m�_�U�S�l�x�����������ûл�ػû������m���~�t�t�}���������ֺ����!����ɺ�������ùïäâÝàù����������������һ������������%�%���������������������'�4�@�M�Q�N�D�4� ��������������������
�����������������<�:�<�G�H�U�a�f�a�[�U�H�<�<�<�<�<�<�<�<�����������������ĽнٽнĽ�������������àØÓÉÓàìñôìàààààààààà��ݹ��������������������� ��������������!�0�8�2�.�!���Z�g�s�������������������s�g�N�A�6�8�A�ZD�D|D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D� \ { L Q 9 ) A ) I ; [ C ( ? \ F Q [ z L  , x A . E 9  M L / , / K S < � ( $ 2 @ S K x ? . : c 7 � L A J b 8 e p ) q C 1 & & @ s P M F D c { V R ? l h    W  �  �  (  <    �  [    >  �    �  D  J  n  �  �  �  �  �     �  �  4    #  �  =  �  Z  �  X    �  d    �    �  &  �  �  `  �  Y  �  �  /  �  �    
  '    �  >  #  �  C  W  t  �  Q  �  r  \  �  Z  �  ?  .  a  O  �  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  M  B  7  +       
  �  �  �  �  �  �  �  {  d  M  6      k  U  ?  (    �  �  �  �  �  �  z  c  L  5     �   �   �   �          �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  y  l  ^  O  @  4  +  "      �  A  L  X  a  d  h  i  i  i  f  d  a  \  V  Q  L  G  =  0  #  �  �  �  �  �  �  �  �  �  o  [  E  )    �  �  �  8  �    e  d  `  Y  O  D  7  *        �  �  �  �  ^    �  �  9     �  �  �  �  �  �  �  �  �  �  `  :    �  �  �  V    �     &  ,  3  9  ?  E  K  Q  W  [  _  b  e  h  i  j  j  j  k  �  �  �  �  y  p  h  `  X  I  8  '    �  �  �  �  �    a  Y  L  @  4  $      �  �  �  �  �  �  �  �  �  �  u  X  <  7  V  p  �  �  �  {  h  S  ;    �  �  �  �  V  �  %  z   �  �  �  �  �  �  |  ^  8    �  �  }  N    �  �  �  n  L  o  �  �  �  �  �  �  z  m  c  [  N  7    �  �  p  (  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  k  _  R  F  q  X  B  4  7  F  O  U  T  N  ?  -    	  �  �  �  �  �  Y  -  @  L  R  Q  J  @  4  '      �  �  �  �  �  �  �  �  p  .  2  4  3  F  9  (    �  �  �  b  "  �  �  ^    
  �  5  "                         %  -  4  <  D  L  S  [  �  �  �  �  �  �  �  �  �  �  |  ^  8    �  i    �  �  E    A  e  �  �  �  �  �  �  �  y  ]  7    �  u  "  �  W   �  i  >  +    �  �  �  �  Q    �  �  G  �  �  �  O  (  �   �  (      "  (    �  �  �  �  �  q  Y    �  Q  �  �  N      +  L  �  �  �  �  �  �  �  r  O  '  �  �  �    z  �          
    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  B  ?  <  9  4  .  (        �  �  �  �  �  �  �  �  x  \  �  �  �  �  �  �  �  �  �  �  {  c  I  -    �  �  �  �  _  �  �    6  L  `  n  q  h  U  ;    �  �      �  $  o  �  �  �  �  �  �  l  J  #  �  �  x  0        �  �  �    [  �  �  �  �  �  �  �  �  b  9    �  �  �  h    �  M  �  )  
    �  �  �  �  �  �  �  �  �  t  _  E  #     �  �  }  M  �    :  F  H  7    �  �  }  A    �  r    �  7  �  [  �  j  �    '  @  G  8  #    �  �  �  �  S  �  T  �  �  �  4  D  8  ,       	  �  �  �  �  �  �  �  o  S  6     �   �   �  �  �  �  �  �  k  U  Q  Z  f  H  "  �  �  U  ,  �  v  �    �  �     �  �  �  �  �  �  _  9    �  �  �  �  r  L  3    D  ?  :  4  /  *  %  #  $  %  %  &  '  $      	  �  �  �  c  \  V  M  @  2  !    �  �  �  �  �  �  z  c  K  4      7  Y  e  m  �  �  �  �  �  �  �  �  �  �  p  8  �  �  3  �  �  �  �               �  �  �  �  Y    �  t  
  �  �  �  �  �  �  �  �  �  �  �  �  �  v  Z  ?  &    �  �  �  �  �  �  ~  v  �  �  m  S  :  %      �  o  6  �  �  u  /  �  i  d  ]  T  G  9  *      �  �  �  �  `  =  "  
  �  �  /  �  �  �  y  `  J  6    �  �  �  �  d  8  	  �  �  i  �  e  L  G  E  D  C  ?  7  *       �  �  �  �  P    �  �  c  �  g  [  O  D  9  -    
  �  �  �  �  �  �  s  `  M  ;  -    j  j  k  l  k  c  \  T  J  ;  ,      �  �  �  �  �  �  �  s  d  P  >  )    �  �  �  �  z  H    �  �  �  �  �  �  �  P  l    �  �  �  �  }  q  ]  A    �  �  �  G  �  �  �  W  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  y  U  2  #    
  �  �  �  �  �  �  �  t  H    |  q  a  L  /    �  �  y  >    �  �  I    �  �  B  �  �  F  A  6  "  
  �  �  �  �  �    ^  G  *    �  �  �  .  S  �  �  �  �  �  �  �  �  j  K  (    �  �  �  �  u  V  7    �  �  �  �  �  �  �  �  �  t  T  -    �  �  q  N    �  �  {  m  Y  A  (    �  �  �  �  o  M  *    �  �  �  �  g  ,  �  �  �  w  [  4  
  �  �  o  9    �  �  \    �  �  N  �  �  �  �      "      �  �  �  O    B  L  +    �  �  7     �  �       �  �  �  d  3    �  �  G  �  t  �  �  s  �  r  y  x  w  E    �  �  �  t  V  "  �  �  T  �  �  �  �  �  �  �  �  �  �  {  g  R  :    �  �  �  }  O  "  �  �  2  �  �  �  �  �  �  �  j  U  <    �  �  �  M     �  �  �  w      �  �  �  �  �  �  �  x  U  -    �  �  f    �  p  "  �  	�  	�  	�  	�  	i  	5  �  �  e  
  �  +  �  <  �  >  �    p  �  �  �  �  �  j  b  S  8    �  �  �    S  )  �  �    �   �  �  	  �  �  �  �  �  �  f  @    �  �  �  U  �  p  �  V  x  �  �  �  �  �  �  �  �  {  r  j  ]  K  6    �  �  �  X    �  �  �  �    r  l  f  Y  F  /      �  �       �  3   �  *      �  �  �  �  �  �  r  O  (  �  �  g  /    �  �  �  t  i  ^  S  H  >  6  .  &              �  �  �  x  O  \  T  L  D  <  2  "      �  �  �  �  �  �  �  z  g  U  C  �  �  �  �  �  �  �  {  h  R  4    �  �  Z  !  �  �  k  ,  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    
    �  �  �  p  +  �  �  ;  �  G  �  �  �  g  -  �  �  �  G    �  u  P  /  �  �  4  �  �  �  �  �  �  �  z  N  %     �  �  �  �  k  Q  9  &      %  A  ^