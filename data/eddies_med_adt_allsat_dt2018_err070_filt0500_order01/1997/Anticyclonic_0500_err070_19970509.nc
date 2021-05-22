CDF       
      obs    B   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?����+       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�E�   max       P�א       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �0 �   max       =� �       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>������   max       @Fk��Q�     
P   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��G�{    max       @vt�\)     
P  +   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @3         max       @O�           �  5d   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�q        max       @�`            5�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �u   max       >Q�       6�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�w.   max       B2#q       7�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�m   max       B1ǘ       9    	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?U�   max       C��       :   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?P	�   max       C�x       ;   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �       <   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          7       =    num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          +       >(   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�E�   max       P �e       ?0   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�e+��a   max       ?��0��)       @8   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ���   max       =       A@   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>������   max       @Fk��Q�     
P  BH   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��
=p�    max       @vt          
P  L�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @&         max       @O�           �  V�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�q        max       @�'            Wl   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         C�   max         C�       Xt   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�-�qv   max       ?������     �  Y|                   
                  %            
   
   �               %         �         F               
      3   	                  *         %                                       *   Z            �      O�_�N[7Ovy_NjdXO��O&LND,:Os�N'�yN�ҎN�_TO�HNPn(O�ÂNL=:O&�NA�|P�אNہ�O�+zO�Q�M�E�O��O���O�!P�@yNX�fN8��P0V�OB��O���O� �N�O���NA�!O��N�v�N���OS�P	0_N���N���P3{�OF@�N�>O��O��OQ��N��&N��9N��N+�CN�7rNN��O+��Ojx�O$JNo�LO���Pa��O�w�M��hO< �O�m#N�g\N�h�0 ż��
��C��u�ě��ě��ě����
��o�D���D���o$�  ;o;D��;��
;ě�;ě�;ě�;�`B;�`B<o<49X<e`B<u<u<�o<�C�<��
<��
<�9X<�9X<�9X<�j<�j<���<���<�/<�/<�/<�`B<�h<��<��<��=+=\)=t�=t�=t�=#�
=#�
=#�
='�='�=0 �=49X=49X=8Q�=T��=u=�o=�O�=��=� �=� �FGN[fjt�������tg]RNF�������������������� 
0<DIPTIF<0#���������������������������������������
	
#/0972/(#


����������������������������������������/.6BDKLB76//////////$),16:BIMOZOKB6)$$$$cdhstv�����tlhcccccc��������������������Z[fgjtx���tg[[ZZZZZZ"/;HTUPMLJH;/"����������������������������������������������������������5[ghUNB5����wz~������������zwwww������������������������������������������������������������qpx~��������������tq���������

�����'().59BNZ[fc][NB5)''�����)BNQB6)����

"#$#
GHNUaglca_UHGGGGGGGG))$+7BOh������th[N,)"!#-0IOU[bb[TD<50" �
#/>HSUYZUOH/#
 lijn{������������{ulLGKO\ahu{����wuh\OLLa\ht��������������ta�yz�����������������#<V^_XNQXKHBE</#���������������������)/565.)������������������������������������������CEJOP[[^ghjlhg[OCCCC������������������������ )8T]]WJ5)"��	"),*.59=52)�����������������������)5BFH@5.)�����������������������������������������NLOO[hmtt����th[VONN���
##),,#

�����')*))#��������������������������������������������������������������������������������BBEILTamoqrstvsmaTHB�����
�������
 #)&#
���������$����������������������������� *4860+$�����������������������������������������������

���������������������������������������������zÇÏãììæÓÇ�z�n�a�U�R�N�W�a�n�u�z���
���
�������������������������������x�������������������������|�x�s�k�h�m�x������������������������������������������������� �$�&�/�$����������ƾƾ���������������������������������������������������������������������������������������������������ŹűūŦŭŹ��ÇÓÙ×ÓÇ�z�w�z�{ÇÇÇÇÇÇÇÇÇÇ���������Ƽʼϼʼ������������������������L�Y�Y�e�f�e�`�Y�L�@�=�<�@�C�L�L�L�L�L�L���ʼҼּ����׼ʼü����������������������������������������������������������T�a�m�w�~�����z�m�a�T�H�;�6�,����;�Tù��������������ùøñùùùùùùùùù���ʾо׾���׾ʾ����������������������n�{ŇŔŗŔōŇ�{�v�n�g�n�n�n�n�n�n�n�n�#�I�`�X�^�c�n�^�M�<����������������
�#E�E�E�E�E�F FF
E�E�E�E�E�E�E�E�E�E�E�E����)�4�(������������������������������"�,�.�F�=�;�"��	��������������������������������������������������������6�h�v�y�h�a�6�)������������ ���������������	�������������������Ź����������������������ŹŸŮŮůŷŹŹ��4�@�M�U�X�T�H�����ݻܻӻ������лܼEEEEEEEED�D�D�D�EEEEEEEE�������������s�r�s�|������������������ʾӾ־Ծ�������s�Y�F�E�J�Z���������������������������������r�o�r�u��`�m�y������y�o�m�`�T�G�?�;�9�7�:�?�N�`���Ľн��߽ѽнʽĽ����������}�z���������	��� �"�$�$�"���	��������������4�A�M�W�[�d�g�c�Z�T�Q�>�4�(����$�%�4�����������������������������������������������ùìà×àìîù������f�s�v�w�s�s�f�d�b�Z�X�M�H�H�M�O�Z�^�f�f�f�j�m�m�k�f�[�Z�X�M�H�K�M�P�Z�a�f�f�f�f�������Ŀ̿ѿԿۿѿĿ���������������������(�<�A�?�7�7�;�D�J�A�(����ۿ���
���#�/�;�<�H�I�H�<�;�/�#���������/�<�H�S�T�H�B�<�/�&�%�)�/�/�/�/�/�/�/�/��(�A�Z�k�j�Z�A�(����ݿѿ¿Ŀѿ����`�m�y�������������������t�m�`�W�T�P�U�`�<�>�<�<�0�/�.�#��#�/�3�<�<�<�<�<�<�<�<Ƨ�����������������ƳơƎ�{�v�}ƎƚƧ�����������ĽʽĽ������������{�z����������'�3�6�?�C�?�?�3�'�"������������S�_�k�l�v�q�l�_�_�S�S�F�C�A�>�<�F�Q�S�S�������������ټ�������������¦°²¿��¿º²¦¤����������������������������������������ìù����������ùìàØÝàäìììììì����������������������������������������������������������������������y�{������U�b�n�{ŇŔŚŗŔŇ�{�n�b�U�I�A�<�>�I�U����!�#�,�#�!���
���������������������~����������������������~�{�s�~�~�~�~�~�:�H�T�Z�W�S�:�-�!�������������!�:���)�4�1�)��
����à�z�a�U�G�A�H�aÇù�����	��/�a�m�t�z�~�z�m�T�H�;�"����������`�e�l�l�n�l�`�[�Z�V�`�`�`�`�`�`�`�`�`�`�������	�����	����������������������DoD�D�D�D�D�D�D�D�D�D�D�D�D�D{DnDaD\DcDo�Z�d�g�p�g�f�c�Z�Q�N�H�D�N�X�Z�Z�Z�Z�Z�Z�g�s�z���������~�s�i�g�c�g�g�g�g�g�g�g�g + h <  E 3 > 1 = L & . I ? L 5 H - H d 4 H P 6 3 2 D M + + C 9 K Z + N d g / Q 4 4 Z Q M X D / K \ Y B 7 J 2 E F t : t l v H 5 9 d    �  �  �  �  e  V  �  =  �  �  T  �  �  j  ~  `  �    D  %    �    d  �  k  n  '  �  s    #  �  V  �    !  �  �  �  �  h  �  B  o  "  �  #  �  �  E  �  r  �  �  l  �  &  )  $  N  �  �  �  c�u�e`B;ě��#�
<���;D��$�  <u:�o;�`B<#�
=\);��
<�j<�o<e`B<�o>bN<�j<���<���<D��=D��=C�=\)>M��<�`B<��
=�E�=C�=Y�=0 �<�h=o=+=���=+=C�=,1=aG�=t�=49X=�t�=0 �=\)=�\)=D��=P�`=]/=,1=D��=@�=e`B=D��=q��=�%=}�=H�9=� �>hs=�{=�+=\>Q�=��=�E�B	��BxB%}{B"5B�B|�B ]B�NB(B�BL?BCbB	l�A�w.B��B;zBK�B��BS�B��B �B��BPJB�+B�%BFB6�B�B�.B&M+B�B)7B2#qB5oB�B�mB��B7]BT�BFB"*B��BL�B��B��Bz�B _�B'fB��B$��B B�B!�BB -�B"t?A�PfB9B$�{B|B%B�[B,بB�BPB�eB�B	��B@B%N�B"�@B�/B�EB ,B�"B0�BȟBAtBAZB	@�A�mB��B,�B@yB�5BB�B�B �B��BBvB@�BٚB?B@�B>"B�B&5vB@B)>�B1ǘB
�B1B�OB�|BH"BP�B� B>�B�(B��B@B��B#<B @|B�zBC�B%=BDvB�rB"<�B  �B"��A��6B0�B$��BQB;�B@�B-�B:�B�B@�BAȨ�A�y@@�ȫ@��+B��A�v?U�A���Aɓ)@�?̊r@��XA��A�4RA΀UAN��A��A�C��A��A[[9Aρ�A׆�A�aA��@��vC�]�AE��AGt�@�(�Ah7�A$E�A[6�A8�l@��SA�#�A@�A?q�At�5A��~A��RA�bAA��AmŶAJB��A!~�?��l@��TAR�A���AH��A���@�Z@�w�A�,�A��|@�O@qWA�OBA�]\A��A��-C��SA�nCA�m�AȁA���@�#
@���B� A�?P	�A�$�A�t�@��?��@�A�l�A�~�A΂<AN4�A�A�0C�xA�U�A[��A�p�A׈�A���A��L@��C�`:AE�AF�K@�%Ah��A#�2A[
uA;�%@�/�A�{�A@A?
�Au �A��MA��\A�q_A��wAl.A�5BɇA"�u?��@���A8�A��AH��À�@��@�ރA�&A�{�@��@v,CA�w&A��A�A���C�ϒA�y$A��               !   
                  &            
      �               &         �         F                
      4   	            	      *         &               	                        *   [            �   	                  !                                       7      #         #         7         +               !      #            )         1         '                                          7   #                                                                     #                        )         #               !                  )         +         #                                             #               O?&4N[7OwXNjdXOZ=O&LND,:O,�N'�yN��N�_TNǾ�NPn(O�ÂNL=:O&�NA�|P�N��Nbl�O9�M�E�N�OO|O�Nл�P �eNX�fN8��O���OB��O��<OG�8N�O���NA�!N�}N�v�Ne=gOA�P	0_NKA�No�P�WOF@�N�>OÅ4O��O&�TN�:qN��9N��N+�CN|EN.�O+��Ojx�O$JNo�LOL�IO���O�w�M��hN���OA�AN�g\N�h  �    �  �  7  �  �  �  i  �  �  /  >  �    �    
�  i  �    �  �  I  *    v  �  �  K  7  �  .    �  �  �  �  �    N  X  !  z  �  �  �  �  �  n  N  �  ;  �  n  �  c  �  �  
=  �    �  C  �  |��㼣�
�#�
�u<o�ě��ě�$�  ��o�o�D��;�`B$�  ;o;D��;��
;ě�=}�<t�<�o<49X<o<�`B<u<��
=���<�o<�C�=�P<��
<ě�<���<�9X<�j<�j=T��<���<�h<�`B<�/<�=o=�P<��<��=��=\)=��=��=t�=#�
=#�
=,1=,1='�=0 �=49X=49X=Y�=�Q�=u=�o=��P==� �=� �XRQW[_gt}������utgdX��������������������#&09<DGC<70#����������������������������������������
	
#/0972/(#


����������������������������������������/.6BDKLB76//////////&),26;BGLOTOIB6)&&&&cdhstv�����tlhcccccc��������������������Z[fgjtx���tg[[ZZZZZZ"/;HTUPMLJH;/"���������������������������������������������������������2>EEA5)��������������������������������������������������������������������������������������������������������������

�����-,35BN[]][XNB5------�������"()$�����

"#$#
GHNUaglca_UHGGGGGGGG6;BO[h�������th[O?86"!#-0IOU[bb[TD<50"�
#/=HJSVXUKH/#
mlln{����������{xsomLGKO\ahu{����wuh\OLLa\ht��������������ta�yz�����������������.&&)/<EHNLHE</......��������������������	)+2))(								��������������������������������������NHMO[ehhhb[ONNNNNNNN������������������������$3BVWPB5)���	"),*.59=52)�������������������������)5AAE:5.)�����������������������������������������MOR[hlst|yth[YOOMMMM���
##),,#

�����')*))#��������������������������������������������������������������������������������BBEILTamoqrstvsmaTHB�����
�������
 #)&#
�������������
�������������������������� *4860+$������������������������������������������������

���������������������������������������������n�zÇÓÚàçàÓÏÇ�z�n�a�Z�U�T�^�a�n���
���
�������������������������������x�������������������������}�x�s�o�u�x�x������������������������������������������������� ������������������������������������������������������������������������������������������������Ź��������������������������ŹŵŰŲŹŹÇÓÙ×ÓÇ�z�w�z�{ÇÇÇÇÇÇÇÇÇÇ���������żʼμʼ������������������������L�Y�Y�e�f�e�`�Y�L�@�=�<�@�C�L�L�L�L�L�L�����ʼּݼݼּѼʼ��������������������������������������������������������������T�a�m�w�~�����z�m�a�T�H�;�6�,����;�Tù��������������ùøñùùùùùùùùù���ʾо׾���׾ʾ����������������������n�{ŇŔŗŔōŇ�{�v�n�g�n�n�n�n�n�n�n�n����0�@�I�N�P�M�<�0�#�
����������������E�E�E�E�FFE�E�E�E�E�E�E�E�E�E�E�E�E�E����������������������������������������޿�"�%�.�3�.�)�"���	����������������������������������������������������)�6�B�L�O�U�O�N�B�6�)������)�)�)�)����������
�
�����������������������Ź������������������ŹŲŲŴŹŹŹŹŹŹ���'�6�>�C�G�E�@�4�'����λûĻɻܻ�EEEEEEEED�D�D�D�EEEEEEEE�������������s�r�s�|����������������Ⱦɾľ�������s�i�\�S�Q�W�f����������������������������������r�o�r�u��`�m�y�}�����y�m�m�`�T�G�B�;�:�9�<�B�P�`���Ľнݽ�ݽнʽĽ������������������������	��� �"�$�$�"���	��������������4�A�M�W�[�d�g�c�Z�T�Q�>�4�(����$�%�4��������������������������������������������� �����������������������޾f�s�v�w�s�s�f�d�b�Z�X�M�H�H�M�O�Z�^�f�f�Z�f�j�k�h�f�Z�M�M�M�N�S�Z�Z�Z�Z�Z�Z�Z�Z�������ĿʿѿҿԿѿĿ���������������������(�<�A�?�7�7�;�D�J�A�(����ۿ���
���#�/�4�;�/�%�#�"������������/�<�H�N�P�H�?�<�/�)�'�-�/�/�/�/�/�/�/�/�(�A�L�Z�c�`�S�A�(����ݿп̿ѿ����(�`�m�y�������������������t�m�`�W�T�P�U�`�<�>�<�<�0�/�.�#��#�/�3�<�<�<�<�<�<�<�<ƚƧƳ������������������ƳƫƚƌƀƄƚ�����������ĽʽĽ������������{�z����������'�3�4�=�>�:�3�'���������������_�g�l�t�o�l�_�Z�S�F�E�@�A�F�S�U�_�_�_�_�������������ټ�������������¦°²¿��¿º²¦¤����������������������������������������ìùþ������þùìàÚßàçìììììì����������������������������������������������������������������������y�{������U�b�n�{ŇŔŚŗŔŇ�{�n�b�U�I�A�<�>�I�U����!�#�,�#�!���
���������������������~����������������������~�{�s�~�~�~�~�~��!�-�:�F�M�S�V�S�F�:�-�!���������ù����������������������ùéÓÈÌÓìù���	��/�a�m�t�z�~�z�m�T�H�;�"����������`�e�l�l�n�l�`�[�Z�V�`�`�`�`�`�`�`�`�`�`���������	�����	��������������������D{D�D�D�D�D�D�D�D�D�D�D�D�D�D~D{DpDoDsD{�Z�d�g�p�g�f�c�Z�Q�N�H�D�N�X�Z�Z�Z�Z�Z�Z�g�s�z���������~�s�i�g�c�g�g�g�g�g�g�g�g , h 5  + 3 > 5 = F & ) I ? L 5 H  0 = / H ) 0 4  D M ' + E = K Z +  d P - Q ! 7 _ Q M Q D 2 8 \ Y B 4 H 2 E F t 9 D l v A ) 9 d  �  �  8  �  >  e  V  w  =  �  �  �  �  �  j  ~  `  G  �  j  �    �  �    �  k  n    �  M  �  #  �  V  �    �  �  �  Z  w  �  �  B  �  "  g  �  �  �  E  �  X  �  �  l  �  �  t  $  N     �  �  c  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  �  �  �  �  �  �  �  �  �  �  �  �  t  L    �  F  �  L  �    �  �  �  �  �  �  �  �  �  {  b  J  +  �  �  �  h  0   �  t  �  �  �  �  �  �  �  �  �  �  �  �  u  A    �  �  ;  �  �  �  �  �  �  �  �  �  {  s  j  b  V  H  :  ,      �  �  �  ^  �  �  �    $  0  6  6  +    �  �  s     �  X  �  _  �  �  �  �  �  �  �  �  �  w  ]  A  &    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  h  Y  I  9  (      �  �  �  a  |  �  �  �  �  �  �  ~  n  Z  A  #  �  �  �  9  �  ]  �  i  m  r  w  z  z  z  y  u  k  a  W  E  /      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  l  T  8    �    �  }  �    y  l  ]  M  =  0  !       �  �  �  �  P    �  �  z  �  �    (  -  /  +  !    �  �  �  L  �  �    `  I  �  �  >  7  1  *  $               �  �  �  �  �  �  �  �  �  �  �  z  p  e  b  \  Z  ^  m  z  t  f  J    �  z    u   �    �  �  �  �  �  �  �  �  x  h  Y  J  <  -      �  �  �  �  �  �  �  �  �  �  x  j  Z  H  5  !    �  �  �  �  �  �            �  �  �  �  �  �  S  #  �  �  �  Q    �  �  J  �  	f  	�  
  
N  
|  
�  
�  
�  
k  
*  	�  	n  	    �  �  �  �  A  M  _  g  ^  N  =  (    �  �  �  �  �  _    �  q    �    "  .  5  4  -     
  ?  �  �  �  u  H    �  h    �    �                  �  �  �  �  �  ~  U    �  �  3  �  �  �  �  �  �  �  	    #  ,  1  7  <  A  ;  2  (      �  �  �  �  !  P  q  �  �  �  �  �  �  Z  #  �  ]  �    e  G  H  B  6  &    �  �  �  �  �  c  7    �  �  B  �  :   O  �  �      (  )  "    �  �  �  �  �  ^  +  �  y  �  }  o  �  1  @  
  �  �  	     �  �  :  �  �  %    
�  	b  �  �  �  v  ^  A    �  �  �  s  F    �  �  �  m  >    �  �  }  L  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    a  �  �  �  �  �  �  �  �  X  '  �  �  H  �  Z  �  �  B  K  F  ;  -      �  �  �  �  �  y  p  q  f  H  %    �  �  -  6  1  #    �  �  �  �  `  "  �  �  1  �  b  �  �  �    �  �  �  �  �  �  �  �  �  �  k  8    �  �  N    �  c   �  .  #        �  �  �  �  �  �  �  �  w  e  T  C  )  
   �                    �  �  �  �  �  �  �  x  ]  G  2  �  �  s  c  Q  ?  -      �  �  �  �  �  �  �  �  �  �  �  ]    G  _  j  �  ?  d  x  �  �  v  X  $  �  T  �  �  �  �  �  �  �  �  �  �  �  ~  o  a  Q  ?  *  �  �  �  �  h  e  c  �  �  �  �  �  �  �  �  �  �  �  �  �  r  M    �  �  R    �  �  �  �  �  �  �  o  [  E  +    �  �  �  �  [    �        �  �  �  y  R  $    �  �  �  �  {  d  R  4    �  	  .  7  A  G  K  M  G  B  9  0  $      �  �  �  �  D   �   �  I  O  S  V  X  W  V  U  R  J  /    �  �  q  2  �  �  8  �  �      !      �  �  �  �  q  7  �  �  U  �  �  �  J  ;  z  z  u  i  [  K  5    �  �  �  �  {  X  8    �  �  r  6  �  �  �  y  q  i  ^  S  G  <  2  (          �  �  �  �  �  �  �  �  �  �  �  �  }  h  O  :  &  !  �  }    n  �  �  �  �  �  �  �    m  T  5    �  �  �  |  J    �  �  a  -  �  �  �  �  �  �  �  �  �  j  L  )    �  �  �  �  x  $  �  �  �  �  �  �  �  �  �  �  �  �  u  N  $  �  {  �  g  �  :  n  d  Z  O  I  C  >  4  '        �  �  �  �  �  �  �  �  N  H  B  :  0  %      �  �  �  �  �  �  �  a  @    �  �  �  �  �  �  �  h  T  E  5    �  �  �  �  �  b  5  �  ]   �  3  7  :  :  9  7  2  -    �  �  �  �  a  4  J  C    �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  m  ]  M  =  B  O  ]  n  L  +    �  �  �    ]  8    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  |  k  V  9    �  �  �  x  H    �  �  c  b  \  R  A  +    �  �  �  k  1  �  �  w  8  �  �  �  �  �    r  f  [  S  K  C  <  8  4  0  )        #  A  `  ~  }  �  �  �  �  �  �  �  �  [  +  �  �  l    �  K  �  B  �  	~  	�  	�  	�  
  
#  
  
  
=  
+  	�  	�  	J  �  Z  �  �  �  �    �  s  s  ]  4    �  �  �  ^  J  >  2    �    '  �  j  �    �  �  �  �  �  �  �  �  �  �  �  �  x  p  h  _  W  O  G  �  v  i  |  �  �  �  p  L    �  �  �  [  +  �  �  K    �    P  S  �  �    ?  @     �  �  ;  �  �  �  �  �  @  �  
!  �  �  v  a  J  4    	  �  �  �  �  �  |  ^  7    �  �  �  |  l  [  K  ;  +    
  �  �  �  �  �  �  �  �  z  i  X  G