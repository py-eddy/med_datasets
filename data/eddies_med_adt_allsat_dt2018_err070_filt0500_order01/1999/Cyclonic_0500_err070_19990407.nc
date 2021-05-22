CDF       
      obs    N   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�� ě��     8  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       Mǳ_   max       P���     8  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��%   max       <#�
     8      effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�        max       @F7
=p��     0  !T   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���G�{    max       @vn�Q�     0  -�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @0         max       @Q`           �  9�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�t�         8  :P   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��x�   max       :�o     8  ;�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��;   max       B06     8  <�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��3   max       B/�~     8  =�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ;�0�   max       C�W|     8  ?0   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >��   max       C���     8  @h   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          J     8  A�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          U     8  B�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          S     8  D   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       Mǳ_   max       P��     8  EH   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�T`�d��   max       ?��e+��     8  F�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ���w   max       <#�
     8  G�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�        max       @F+��Q�     0  H�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���G�{    max       @vn�Q�     0  U    speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @(         max       @Q`           �  aP   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�          8  a�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         FI   max         FI     8  c$   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�Fs����   max       ?�Ʌ�oiE     `  d\   
               7         	      ;         
                              "      
                  #         :   ,   &         "                     	               0   $   	   	      .                  I      
            
   !            !   	         N��DO�%N@-CN��LP4t�P���NZ�NmgN�4�N���P�P�hO��O5�fPebNF�6NzON��uMǳ_O�EN^�O���N9�O���N�2XN��]O�NúZN��>N��KOuS�O��.O��NdoP���O�	�O���N��O(&�O��OD�sNb[N�ҊOn�nOZOeN��N~�N(tO�~O8�xO���O��N8NC�O�kP��OJ!M��N˞O�g&NY�JPc8iN%��N�oO��CN��sO�N��_O�'wOd��O�O��N��3N���N��9O�ICN�o�<#�
;�o;D��;o%   ��o��`B�t��D���D���D���D���T���T����C���C���C���t���t����㼴9X��9X��j��j��j��j��j���ͼ�����������������`B��h��h��h��h��h���������o�o�+�C��\)�\)��P��P��w��w��w��w�#�
�#�
�#�
�#�
�#�
�#�
�#�
�#�
�',1�,1�0 Ž0 Ž@��@��H�9�H�9�H�9�aG��aG��ixսu��%��%��
#(.#
�����������
#%*0+#
�������������������������������
������������
#+HWRH#
��������#<j�������nU<����8BFORZ[_[OB>88888888��������������������-/2;HMORH;0/--------��������������������(,6[hntvvx{��u[B6*"(uu{��������������squ�������������������� )69BOUXYWXVOB6.)#<PYany�����mU</##hnsz�����zndhhhhhhhh��������������������+/<HSKIH<2/,++++++++��������������������<FHUanz����znaUH<55<����������������������������������������RT^adeeaZTQQRRRRRRRR�������������������������������Q[^gst{|}tgc[SQQQQQQegkqt���������tlgde�����		���������������������~yw46BKOW[fhthd[OKB=:64qzu���������������tq
6CSX[[YTOC6�:==?@CHTVY\^]ZUTOH;:LTU_adda]TTOLLLLLLLL)B[t���������te\B5))7=Hanz������znaUH=87���������������������������������������������������������������#25.#
���������������������������������������������������
 
�����u������������������u��������
����16BCO[^fhb[ODB;76511������������������������������������������������������������),6BO[db][ROGB66)Y[gt��������tg_[QPYY��������������������������������������
#%#

�����������������������������������������)5BP]`[H5)�������$),*'!���)5676)ABIO[hoojha[VOB8AAAA	#0DMNC<60# 
	*0<@DG<0-%**********�#0IQKCG<5.
��?BBNV[]][YNB????????�'%# ����������������������������������������������������������������������� �������������������������)5BR[fgtwsg[NB5,&!���� " ����������"/9A5) �����"#/<GE?<9/&##""""�����������������������������������P]g�������������g`WP���������������������������������������üǼȼ����������������g�[�Q�B�>�B�J�N�[�t�t�gE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��������������������������������������1�!�
����(�:�M�f�p���������f�X�[�A�1������n�W�(����A�����������������	���a�V�U�N�U�a�d�n�q�x�r�n�a�a�a�a�a�a�a�a��������������������������������������������������������������ؿ������Ŀƿѿݿ����ݿѿĿ������������G�;�/����$�.�;�T�`�y�������������m�G�׾����������w�n�Q�E�Z����ƾξ׾�	����àß×ÔÖÑÓàãìù����������úùìà�6�/�'�$�"����"�/�;�H�T�^�]�[�T�H�=�6�����������~���������������������������;�4�/�(�/�/�;�F�H�H�J�H�;�;�;�;�;�;�;�;�N�M�C�I�N�Z�]�g�k�q�g�Z�N�N�N�N�N�N�N�N�6�2�/�-�6�B�O�O�[�O�H�B�6�6�6�6�6�6�6�6�)�'�'�'�)�.�6�8�7�6�)�)�)�)�)�)�)�)�)�)�L�5�,���&�/�A�Z�g�h�n�}�����������g�L�ɺ��ºȺɺҺֺ����ֺɺɺɺɺɺɺɺɿ��y�i�^�V�T�O�T�m�t����������������������ƻ��������������������������������������������������ĿĽķĿ�����������������񾾾������ʾ;׾޾�����������׾ʾ������������������ĿѿֿܿѿʿĿ������������C�<�6�*�&����*�6�C�M�O�\�e�b�\�V�O�C��
�	���������	���"�/�4�6�/�"����������������������������{�x�������������������������������������������r�e�f��������ʼּۼ޼׼˼ȼ��;�.�	���־ݾ���	��"�.�5�<�A�B�K�G�;���������s�g�W�T�Z�g�s����������������������"�%�/�;�H�;�0�/�"��������®¢¦²����/�U�[�O�<�
������®�������|�u�x�������ùϹ����߹ҹù����x�V�T�P�T�_�x�������������̻ʻû������x�U�K�H�@�<�9�<�=�H�U�U�a�g�n�n�n�h�a�]�U�A�@�F�@�A�N�Z�g�s���������������s�g�N�A�����������Ŀѿݿ�����!�5�:�(����ѿ������������������нݽ�����ݽ׽нĽ������������������������������������������0�&�$���	����$�0�=�?�F�I�K�I�=�0�0�лɻû�������u�u�x�����������ûллһ�ŠşŔŇŃ�{�x�u�x�{ŁŇŔŚŠŢťţŦŠ�����������������Ľн۽۽нϽĽ����������ϹϹٹݹ߹ڹܹ߹������� ����߹ܹ�¦¦ª²·¿¿¿²¦¦¦¦¦¦ƚƖƎƁ�wƁƎƚƧƲƧƞƚƚƚƚƚƚƚƚ��������������������������������������������������	��������	�����׾ʾ��������ʾ׾����	����������D�D�D�D�D�D�D�EEEE*E0E*E'EEEED�D�E7E3E7E?ECENEPETEYEPEPECE7E7E7E7E7E7E7E7ÓÒÇ�z�w�zÇÓàæìõìàÓÓÓÓÓÓ�U�H�<�/�,�&�)�/�<�H�U�V�a�i�n�s�n�a�\�U��ñäÙÑÒÖàìùþ�������������Ҿs�o�f�Z�X�M�A�8�A�O�Z�f�s���������w�s�ܹ׹ٹܹ������ܹܹܹܹܹܹܹܹܹܹϹιù������ùϹܹ������ܹϹϹϹϽ����������������ݽ�����нʽ½��������������&��������������_�:���$�:�F���л��������ܻ��x�_�5�)�)�����)�1�5�;�9�5�5�5�5�5�5�5�5�����������(�+�(�����������������������������������*�4�7������������*�6�?�C�E�C�<�6�*�����r�e�V�K�L�Y�~�������ɺ���к��������r�L�A�@�L�Y�e�m�r�~�����~�z�r�e�Y�L�L�L�L�W�N�^�c�s�������������������������s�g�W�����������������������������������������G�:�.�!��!�.�:�S�`�l�|�����}�y�l�`�S�G�[�P�J�I�P�h�tāčĘĔĚěĕĔčĀ�t�h�[E�E~EuEzE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�����������~����������������������������#� ���
�	�
��#�0�;�<�A�A�<�0�#�#�#�#ĚčąāāčĚĦĳĿ��������������ĿĳĚ������������$�.�0�1�0�$����������� < - 8 @ L | @   $ u ' | D 8 L J O B y 6 0 M \ S [ 8 6 b U a ^ ; | � G ) + 8 F H $ s E _ P S � T � K 6 % M j � ? A B q < ( > l J C P & i [ X ] 5 R @ , O - D  �  �  W  �    	m  �  t  �  �  x  "  C  �  �  `  u  �  ;  �  {  ?  v  g    �  .  �  �  5  C  �  �  �  �  �  
    �  F  �  <  �  \  |  3  �  �  s  d  �  �  L  k  x  V  �  ]  M  �  T  b  �  ^     +  �  K  �  7    +  w  �    �  �  �:�o�������
���
��`B�q������u���
��o��\)��`B�o��j��w���
��1��`B��1�49X��h����/�e`B�\)�+�\)���+��P�49X�u�,1�o�� Ž����+�T���0 Ž�o�D���\)�0 Ž�%�T���'49X���,1�P�`�q����-���P�@��D����%�� Že`B�8Q�aG���C��0 Ž�xս8Q�T������H�9��O߽ixս��T������㽟�w��9X��+�����^5��O�B$�B�kB��B�B�"B%��B��B�A���B ��B�qB }ZB�DB qB̍B%oB��B�iB��B�%B!��B*�A���BBB	=�B
�A��;B
�B�LB��B06A���A�dB	�Bn8B�B!?qB>%B_�B!kFB��B��B DB��B�0B ��B�9BXYBQB	�)Bt?Bc�B$�B��B��B�BK�B�B7|B%�B&"B%E/B:PB:FB�B�VBX�B#�B)B�B�uB�>B;�B��B�pB
Z�BAyB$� B;~B�~BA'B6�B(��BE[BܒA���B �aB�DB!@IB�]B@.BH�B?�B�]B�=B|jB��B!�ZB*�NA��nBH�B<�B	A$B	��A��3B
��B��BB/�~A���A��QB
@
B>B?�B!?�BDB8�B!B@B�B7�B >�B��BӏB �%B��B}�B�_B	��B��BFvBA5B�DB��B��B��BʿB7�B%K�B%�vB$ǤBAwB@/B�/B��B�[B#?�BB��B��B�AB?�B7 B�dB	�UB8�@��{A�
+C�W|A���A;�;A�4�A���@��ZA䕋A|G�AiДAK�{Ã�A�E�A���A���A���A�;A֛�A���@9�An�"B�IA��kAS��Ax` B �A���A�� @�(@��A\.A��A�B8A�x�;�0�@��AŪ�A��NA~&ZA'�BtB
7@�
hA��A%�?Q�A��SB�A�0AY��AU��C�X�C���Aʺ�Aĭ�A�ʕAA�>�_�>�#8A#m�A3 	@�[hA��VA0d	A���A�vo@��?���A�A�u�A�wA�K�C��@��HA���A�4YB	 �@�@#A�}C�XA���A<��A��AƁ�@��A�w�A~�;Ak
�AKA̱�A���A��TA�z>A�t*A���A��)A���@8��And4BE�A��AS=zAwgB<�A�G�A��e@�N�@�qA[o{A�OA��<A�_�C���@�.\A�zA�q�A|��A'"�B�\B	�L@��8A�WA&�?/� A�B�A�UAZ@AU�C�V�C��4Aʉ%AĐ$A�z2AB�5>�|�>��A"�A2�i@�9A��.A0�jA��vB �@ѕ?�/�A��A��UA$�A�~"C��@�_	A�A߅3B�Z   
      	         8         	      ;                                       "                        #         :   -   &         #                      
               1   $   	   	      /                  J                     !            "   
                        3   U               )   3         )               #                                    #         ;   #   #         %                                                   )                  ;               '      #                                          S                  3         )                                                            )   #                                                               )                  !               '      #                        N��DO>��N@-CN��LO���P��N=��NmgN�4�N>��O��"P�hN���O5�fPebNF�6NzON��uMǳ_O:r�N^�OR �N9�OeqNxJ�N��]N�G�NúZN��>N��KO\�Og�zO�NdoO��O�	�O�,`N��O(&�O�P�O�eNb[N�ҊO(�JN��vOeN��N~�N(tOw�OͧOw�/Nۦ�N8NC�O�kP��N���M��N-�qO'F?NY�JO�F�N%��N�)OE�N��sO�N��_O�'wOd��O;��O��N��N���N��9O�ICN~'Y  �  \  �  ~  [  �    0  �    .  w  d  �  �  �    r  �  �  t  �  �  c  0  [  =    �  �  =  i    T  �  �  l  �  �  �  �    �  �    m  )  f  e  �  �  [  
N  �  �  �  �  g  �  [  �    �  �  �    w  N  J  3  �  �  U  	|  �  B  �  {<#�
�o;D��;o�#�
���
�o�t��D���T����`B�D������T����C���C���C���t���t���`B��9X�ě���j��/������j�ě����ͼ���������/�t���h��h�P�`��h�t���h�����+���o���\)�C��\)�\)��P���'L�ͽ,1��w�#�
�#�
�#�
�0 Ž#�
�<j�D���#�
���w�,1�49X�@��0 Ž@��@��H�9�H�9�e`B�aG��}�ixսu��%��o��
#(.#
������������
#%),(#
������������������������������
�����������
#/GMLH</#
 �����#<k�������nU<����9BKOQY[][OB?99999999��������������������-/2;HMORH;0/--------��������������������2:BO[hoovxth[OB=4,-2uu{��������������squ�������������������� )69BOUXYWXVOB6.)#<PYany�����mU</##hnsz�����zndhhhhhhhh��������������������+/<HSKIH<2/,++++++++��������������������;?DHMUanswuongaUHE<;����������������������������������������RT^adeeaZTQQRRRRRRRR����������������������������������Q[^gst{|}tgc[SQQQQQQfglst���������tmgeff�����		���������������������~yw46BKOW[fhthd[OKB=:64s|���������������ts *6CJQRROD6*HTUX[]\YUTNH<>>??ADHLTU_adda]TTOLLLLLLLLW[t�����������tjg_WW7=Hanz������znaUH=87���������������������������������������������������������������!$
����������������������������������������������������
 
�������������������������������������16BCO[^fhb[ODB;76511������������������������������������������������������������")/6BOX[ba[QOJB86)&"R[\gt��������tga[TRR���������������������������� �����������
#%#

�����������������������������������������)5BP]`[H5)�����&&$������)5676)NO[hjih[OHNNNNNNNNNN#0;<DFEA<90##*0<@DG<0-%**********#068/-,)#?BBNV[]][YNB????????�
""�������������������������������������������������������������������������� �������������������������)5BR[fgtwsg[NB5,&!��� ��������"/9A5) ����� #(/:<@<<6/#       �����������������������������������P]g�������������g`WP���������������������������������������üǼȼ����������������t�g�[�X�N�I�D�N�U�[�g�t�tE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��������������������������������������(���%�,�2�A�M�Z�_�i�m�u�w�f�Z�M�A�4�(�������n�X�(�����N��������������� ��	���a�W�U�R�U�a�h�n�p�w�q�n�a�a�a�a�a�a�a�a��������������������������������������������������������������ؿĿ��Ŀ̿ѿݿ����ݿѿĿĿĿĿĿĿĿĿG�;�6�4�0�4�G�T�m�y���������������m�`�G�׾����������w�n�Q�E�Z����ƾξ׾�	����àÛÙÝàìùù��þùìàààààààà�6�/�'�$�"����"�/�;�H�T�^�]�[�T�H�=�6�����������~���������������������������;�4�/�(�/�/�;�F�H�H�J�H�;�;�;�;�;�;�;�;�N�M�C�I�N�Z�]�g�k�q�g�Z�N�N�N�N�N�N�N�N�6�2�/�-�6�B�O�O�[�O�H�B�6�6�6�6�6�6�6�6�)�'�'�'�)�.�6�8�7�6�)�)�)�)�)�)�)�)�)�)�g�Z�N�F�A�9�7�9�A�N�Z�g�h�s�y�������s�g�ɺ��ºȺɺҺֺ����ֺɺɺɺɺɺɺɺɿ��y�m�k�a�Y�X�`�m������������������������ƻ���������������������������������������������������������������������������̾ʾƾ��ʾ׾�������׾ʾʾʾʾʾʾʾʿ��������������ĿѿֿܿѿʿĿ������������C�?�6�*�)� �*�0�6�C�I�O�\�a�a�\�U�O�C�C��
�	���������	���"�/�4�6�/�"����������������������������{�x�����������������������������������������y�r�g�r����������ʼּܼռɼ�����	���������	��"�.�4�8�6�.�(�"��Z�W�Z�g�s�������������������������s�g�Z����"�%�/�;�H�;�0�/�"��������¿¸¯±¿�������
��#�<�K�J�B�/�
����¿�������|�u�x�������ùϹ����߹ҹù������x�^�W�]�l�x���������������»����������U�K�H�@�<�9�<�=�H�U�U�a�g�n�n�n�h�a�]�U�A�@�F�@�A�N�Z�g�s���������������s�g�N�A�ÿ��������Ŀѿݿ�����������ݿѿý����������������Ľнݽ���߽ݽнĽ��������������������������������������������0�&�$���	����$�0�=�?�F�I�K�I�=�0�0�����������z�}���������������ûʻƻû���ŇŅ�{�y�w�y�{ńŇŔŗŠŠţŢŠŠŔŇŇ�����������������Ľн۽۽нϽĽ����������ϹϹٹݹ߹ڹܹ߹������� ����߹ܹ�¦¦ª²·¿¿¿²¦¦¦¦¦¦ƚƖƎƁ�wƁƎƚƧƲƧƞƚƚƚƚƚƚƚƚ����������������������������������������������������	��������	�����׾ʾľ����ƾʾ׾����	���
�������D�D�D�D�D�D�D�D�EEEEE$EEEEED�D�E7E3E7E?ECENEPETEYEPEPECE7E7E7E7E7E7E7E7ÓÒÇ�z�w�zÇÓàæìõìàÓÓÓÓÓÓ�U�H�<�/�,�&�)�/�<�H�U�V�a�i�n�s�n�a�\�U��ñäÙÑÒÖàìùþ�������������Ҿf�[�Z�U�M�M�M�Z�f�s�~�������s�f�f�f�f�ܹ׹ٹܹ������ܹܹܹܹܹܹܹܹܹܹù¹����ùϹҹܹ۹Ϲùùùùùùùùùý������������������ĽνнԽѽнĽ����������������&��������������l�_�Y�Q�T�_�x�����ûлڻܻ߻û������x�l�5�)�)�����)�1�5�;�9�5�5�5�5�5�5�5�5�������������������������������������������������*�+�,�����������������*�6�?�C�E�C�<�6�*�����r�e�V�K�L�Y�~�������ɺ���к��������r�L�A�@�L�Y�e�m�r�~�����~�z�r�e�Y�L�L�L�L�W�N�^�c�s�������������������������s�g�W�����������������������������������������S�G�<�.�$�)�.�:�G�S�`�l�r�y�{�{�r�l�`�S�[�P�J�I�P�h�tāčĘĔĚěĕĔčĀ�t�h�[E�E�E�E|E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�����������~����������������������������#� ���
�	�
��#�0�;�<�A�A�<�0�#�#�#�#ĚčąāāčĚĦĳĿ��������������ĿĳĚ������� �	���$�,�$�$��������������� < & 8 @ L } C   $ `  | - 8 L J O B y 7 0 > \ K N 8 2 b U a \ $ v � _ ) $ 8 F *  s E Q = S � T � B 4  = j � ? A ; q D  > f J % J & i [ X ] 8 R ; , O - S  �  �  W  �  j  	P  n  t  �  }  m  "  �  �  �  `  u  �  ;  �  {  �  v  �  �  �  �  �  �  5    �  �  �  �  �  a    �  =  L  <  �  �    3  �  �  s  >  f  �  �  k  x  V  �  �  M  I  ]  b  �  ^  �  �  �  K  �  7    �  w  �    �  �  �  FI  FI  FI  FI  FI  FI  FI  FI  FI  FI  FI  FI  FI  FI  FI  FI  FI  FI  FI  FI  FI  FI  FI  FI  FI  FI  FI  FI  FI  FI  FI  FI  FI  FI  FI  FI  FI  FI  FI  FI  FI  FI  FI  FI  FI  FI  FI  FI  FI  FI  FI  FI  FI  FI  FI  FI  FI  FI  FI  FI  FI  FI  FI  FI  FI  FI  FI  FI  FI  FI  FI  FI  FI  FI  FI  FI  FI  FI  �  �  �  �  �  �  �  �  �  �  �  x  \  :    �  �  i  *   �    5  J  Y  \  Y  O  >  &    �  �  \    �  *  �  �  f    �  �  �  �  �  �  �  r  r  t  m  ^  M  4      �  �  �  �  ~  q  c  V  I  ;  .  !            �  �         �  �  �  	           8  Y  I  0    �    (    �  �  �  �  >  �  �  w  J    �  �  �  o  W  -    �  �  N  �  ^  %  �   s  �               
    �  �  �  �  �  �  w  F    �  �  0  -  *  &        �  �  �  �  �  �  �  �  u  a  9    �  �  �  x  l  ^  P  >  ,      �  �  �  �  �  v  R  -    �                                           �  �  �    &  -  *    	  �  �  �  t  J    �  6  �  �  �  w  f  T  C  -    �  �  �  }  R  1  "  �  �  �  �  �  o  .      J  L  J  B  d  R  <    �  �  �  y  F    �  Y  �  P  �  z  n  _  P  >  -      �  �  �  �  �    N     �   �   g  �  �  �  �  �  {  k  W  @  "  �  �  �  �  �  c  *  �  |  m  �  �  �  �  �  �    w  m  c  Z  P  F  =  3  *                     �  �  �  �  �  �  �  �  �  �  �  �  |  u  m  r  a  O  6      �  �  �  �  k  M  ,    �  �  �  �  c  @  �  �  �  �  �  �  �      '  8  I  Z  s  �  �  �    D  m  A  ^  o  �  �  �  �  �  �  �  �  �  �  t  W  %  �  p  �  z  t  r  q  k  ]  O  ?  /      �  �  �  �  �  |  Z  7    �  �  �  �  �  �  �  �  �  �  �  �  w  \  >    �  �  �  ]   �  �  �  �  �  �  �  }  w  r  l  i  i  h  h  g  b  [  U  N  G  b  O  b  K  '     �  �  o  ;    �  �  �  :  �  Z  �  M  �  �    #  +  .  0  .  (      �  �  �  w  :  �  �  o  '  �  [  E  .    �  �  �  �  �  f  J  0    �  �  �  �  �  �  �  4  :  =  ;  5  +      �  �  �  �  �  �  s  W  7    �  �      	    �  �  �  �  �  �  �  |  ]  ;    �  �  �  ]  -  �  �  �  ~  r  f  W  F  5  #    �  �  �  �  O    �    4  �  �  �  �  }  h  S  >  +    �  �  �  �  �  �  �  o  I  "  *  <  2  )      �  �  �  �  �  �    ,  #    �  �  �  �  �  �    F  _  g  i  e  W  ?    �  �  �  Q    �  s    �        �  �  �  �  �  �  z  s  m  e  X  H  3      �  �  T  I  >  2  '        �  �  �  �  �  �  �  �  �  ~  m  [  �  �    H  l  �  �  �  �  �  X    �  Y    �  �  �  �  �  �  �  �  �  �  k  M  ,    �  �  �  �  \    �  D  �  #    -  P  e  l  k  a  M  4    �  �  �  n  ;  �  J  �  l  g   �  �  �  �  z  Y  .    �  �  �  V  !  �  �  ^  �  y  �  +  {  �  �  �  �  �  �  �  �  �  ~  p  ^  I  /  
  �  �  �  �  h  X  x  �  �  �  �  �  �  y  ^  <    �  �  X  
  �  H  �  �  �  �  �  �  �  �  �  �  �  n  P  .  
  �  �  �  V    �  U    �  �  �  �  �  �  ~  h  R  ;  $    �  �  �  �  �  ~  f  �  �  y  l  _  R  M  J  D  <  1  '        �  �  �  �  �  q  �  �  �  �  �  �  �  {  Y  4    �  �  �  �  D  �  (  �  �  
        �  �  �  �  Q    �  �  R    �  �  A  �  y  m  ^  O  B  7  -         �  �  �  �  �  �  �  �  �  �  �  )    �      $  )  0  7  B  P  `  v  �  �  �  �  �  �  �  f  ^  V  N  G  ?  7  .  $         �   �   �   �   �   �   �   �  e  S  @  .      �  �  �  �  �  l  Q  8      �  �  �  u  �  �  �  �  �  x  i  X  H  6  "    �  �  �  i  >    �  �  �  �  �  �  �  �  �  �  �  �  �  W  &  �  �  }  <  �  �  C  �    ?  R  [  Z  N  9    �  �  �  ?  �  �    �  �  �  i  	�  	�  
?  
  	�  	�  	~  	D  	  �  z  &  �  i  �  �    �  �  �  �  �    o  ]  M  C  8  ,     �  �  y  E    �  �  j  0  �  �  b  C    �  �  �  }  V  .    �  �  �  u    r  �  �  8  �  �  |  a  E  +    �  �  �  p  =    �  �    �  �  0  @  �  �  �  �  �  �  p  ?    �  q    �  z    �  �  H  B   �  M  W  `  e  _  L  6      �  �  �  �  D  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  s  _  L  9  	    '  7  B  L  T  Y  X  P  B  0      �  �  �  �  Z    �  �  �  �  �  �  �  �  �  �  z  i  ]  O  >  (  �  �  [         �  �  �  �  �  �  �  �  �  �  �  r  [  D  -     �   �  �            M  �  �  �  �  b  $  �  �    �  �  C  �  �  �  �  �  �  �  �  �  �  �  �  �  r  h  d  a  ]  Y  U  Q  �  �  �  �  �  �  �  �  �  �  �  �  ~  c  L  6  "    �  �  �  �            �  �  �  �  �  m  B    �  �  o  6  
  w  w  w  v  r  l  f  ^  S  I  <  +      �  �  �  �  �  �  N  4  &    �  �  �  y  V  @    �  �  �  u  ,  �  r     �  J  A  8  .  $      �  �  �  �  f  8  	  �  �  �  g  @    3       �  �  �  �  e  ;    �  �  \    �  S  �  [  �    �  �  �  �  �  }  o  _  L  3    �  �  �  u  <    �  p  �  J  m  �  �  �  �  �  �  w  S  -    �  �  �  �  X  0    �  U  1    �    �  �  �  �  d  8    �  �  �  m  E  �  �    	  	-  	R  	n  	|  	z  	r  	a  	@  	  �  �  E  �    f  �  �      �  p  _  L  9  &    �  �  �  �  �  e  @    �  �  �     �  B  =  7  2  ,  #        �  �  �  �  �  �  �  �  b  ?    �  �  �  �  �  �  �  �  �  �  r  N  *  �  �  ^  �  ;  Y   c  z  {  {  {  q  e  Y  I  7  &      
  �  �  �  �  �    `