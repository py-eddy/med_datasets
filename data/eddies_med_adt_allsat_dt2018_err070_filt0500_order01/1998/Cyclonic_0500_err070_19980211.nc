CDF       
      obs    J   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?��+I�     (  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N<   max       P��     (  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ����   max       <��
     (  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?=p��
>   max       @Fh�\)     �  !$   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �������    max       @vj�G�{     �  ,�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @         max       @Q�           �  8D   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @���         (  8�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �%�T   max       <#�
     (  :    latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��    max       B0f9     (  ;(   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A���   max       B05     (  <P   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       =�/�   max       C�=     (  =x   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >@�   max       C�&�     (  >�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          X     (  ?�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          G     (  @�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          A     (  B   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N<   max       P�#G     (  C@   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�5?|�i   max       ?�$xF�     (  Dh   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��
=   max       <�t�     (  E�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?\(�\   max       @FJ=p��
     �  F�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���Q�    max       @vi�Q�     �  RH   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @&         max       @Q�           �  ]�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�;          (  ^l   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         D�   max         D�     (  _�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�hr� Ĝ   max       ?�$xF�        `�   *   
         U               ;      )   )   4            /      1                     	                        0      ;                                              <      
         	             X   
   
   
                              O�>�N���OP�N���P��OVyN/"N[�Of�O���O�@MP\�P�mPuN3MN�YBN�QnP��OUP-�PO�kN��O�~�N��N:�bO|��N��>N<O���O_bNe�O�ĎNV-RN�sO�w�NV2�P&?vO���O��Nv��O�BO�C�Nb>5N*VO��OZ]uO@�!N���O5�$OCN�NT�lP)�hNF�N���N���ND[�O
�VN�YOH��O'��P�4N1x�N��LN�t�Os-O��SN��Ok�iO'�UOp�_O�4N�XN�N�H/<��
<��
<�o<t�<t�;�`B;��
;o��o��`B��`B�u��C���t���t����㼛�㼛�㼬1��1��9X�ě����ͼ�/��/��/��/���������o�o�+�+�\)�\)�\)��P��P�#�
�,1�0 Ž0 Ž49X�49X�8Q�<j�@��H�9�H�9�T���aG��aG��e`B�ixսu��+��7L��\)�����������㽛�㽥�T����9X��9X��Q���ͽ��ͽ�������
#<FWW_UNLH</
	)6686-)						���

�������������������������������#In�������^0��BHKTajmvwvumiaTH?;=B���

�������������#&0<EC<20#������������������������������������~�����������������~~)-..6Qh�������thO6*))6BTbkx���hB61);Oaz��������ujb_Q<4;Zanz����zncaZZZZZZZZ��������������������/;<HUXajfaYUKH@<23//���*6Ci��~uhO6*��GHUanz������zrnkaYUGUn������������nUKKMU!)5N[gt����tg[N53-)!KOTZ[`hpha[OKKKKKKKK"0<Ubdc`UNKIG@<#/<DHHIH</#-6<BKOSODB62--------19BO[hntupmkh[OF@>61~����������yz{~~~~~~	$%										������������������������������-59BFNTSNB95--------�������	
	��������#0<FIIIB<90#"gmz|������zzmjefgggg����������������������������������������N[et�����������tgRKNloz�����������znghkl
")35BDGHDB=5)
������������������������������������������������������������;<HHRU^]XUQH<7;;;;;;����������������������������������������knz�������������znjk
)5BEBA=5)%
#%,/<CGD<7/#���������������������������������������������� ������������������������������cnrz~����~zncccccccc��
#(+,# 
	�����55>BNY[aca[WNMB?7555��������������������RU^bnputtrpnhb[URLLR��������������������#'/<ADD?<8/#������������������������������������������������������������()5>BLB:5)U[gt����{tga[QUUUUUUz���������������wvzz������(*&������������������������������|zwvx��������������������).5BQ[gt{~tmc_WNB5()���������������������������������������������������������������������������������#�)�6�B�O�Q�[�h�k�q�m�\�O�B�6�)��ֺӺκѺպֺ���������ֺֺֺֺֺ�ŔŒŌœŠŹ��������������������ŹŭŠŔ�U�Q�H�B�<�6�1�<�H�U�[�a�c�d�a�[�U�U�U�U�����������Z�D�4�.�3�A�Z�����������	����ƵƳƮƮƳƶ�������������������������������������
����
���������������������/�%�"���"�/�3�;�?�@�;�/�/�/�/�/�/�/�/àÕÓÌÆ�z�v�{ÇÓßìó������ÿùìà�C�6�*�������*�6�C�O�\�a�g�k�h�\�C���u�j�f�d�l�x���������ûл׻Իͻ��������"���ʾ����׾۾پ�����;�b�k�m�c�T�"���}�_�[�������л����"����л���������������������(�5�B�Z�g�s�s�Z�5���H�G�C�C�F�H�P�U�Z�Z�U�N�H�H�H�H�H�H�H�H�������$�.�0�=�=�=�7�0�(�$�����#�������#�/�3�<�F�<�:�<�>�<�/�#�#�;�.�"��������.�;�G�Q�g�h�_�`�Y�X�G�;�����������������������������������������������z�o�w�������̺������Ϲù��������������	��� �!� ����	����Ϲǹù��������ùǹϹԹ׹ϹϹϹϹϹϹϹϽ������x�v�z�����������Ľн�ݽнϽ������M�F�B�C�K�M�Z�`�f�g�j�o�m�g�f�Z�M�M�M�M������������������������������������������~�����������������������������N�L�N�N�S�Z�g�k�r�s�|�s�g�Z�N�N�N�N�N�N����	���'�/�'�'�����������3�2�?�C�M�Q�e�r�~�����������~�h�Y�L�@�3�������¾ʾ׾��������	�	�����׾ʾ����нƽĽ½Ľɽнݽ��ݽؽнннннннннǽĽ��������Ľݽ������������о������� �(�)�(����������������������������	����������������������
��	��������;�T�a�g�\�T�H�.�"�ììâìöù��������úùìììììììì����ŭœŇ�t�{ňŔŭ�����������������"�����������	��"�/�;�C�G�I�N�H�;�/�"�����������������������������������������t�k�k�h�t��t�t�t�t�t�t�t�t�t�t�m�`�W�`�`�m�y�����������������������y�m�	����վӾ׾���	�"�3�;�G�K�S�G�3�"��	���������������������������������������������������������������������������������A�>�5�5�8�A�Z�g�s���������������~�g�Z�A�������������Ŀѿݿ�����������ԿѿĿ��<�0�"����#�,�0�<�H�S�U�]�b�e�b�U�I�<D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�ƎƊƅƎƏƚơƳ����������������ƳƧƚƎ�����z�����������ĿǿѿڿۿѿĿ�����������
�����������������������r�g�W�V�^�r�~���ֺ���������ɺ������!���������!�#�&�!�!�!�!�!�!�!�!��������!�-�:�<�:�5�.�-�!��������������������������
��
������������)�"�)�6�;�B�O�T�Y�O�B�6�)�)�)�)�)�)�)�)�����������������ûлܻ�����ܻлû��Y�Y�X�Y�Z�_�f�o�r�}�����������r�f�\�YEuEjEqEuE}E�E�E�E�E�E�E�E�E�E�E�E�E�E�Eu�I�H�E�J�U�Z�b�l�n�{ŇŒŇņł�y�n�b�U�I�B�6�)��	����)�6�O�[�i�m�t�{�v�j�[�B�������������������������������������������������������������������� ��������"�"���������²©¦¡ ¦ª²¿��������������¿²²�������������}�|���������ĽԽ�ؽڽѽĽ��G�C�@�G�J�S�Y�`�l�u�p�l�`�S�G�G�G�G�G�GčĆĄČĚĦĳĿ��������������ĿĳĦĚčĿĺĶķĻĿ��������������������������ĿÇ�z�o�c�U�F�<�;�<�H�K�U�n�zÓàâàÓÇ�ּʼ������üʼټ�������������ּ������������������������������������������#��������������@�8�4�2�4�@�G�M�Y�f�k�l�h�f�Y�M�@�@�@�@ 8 T T V T  r U G & 2 J w 7 } 0 D G s ? ` _ 8 G A 4 @ O q ] G / b g J D 7 = g = L V n 5 f m ^ p 9 q X O � d  e K E F c  k H N J 0 % 8 4 � Y y P 3  s  �  �  �  =  �  v  �  �  w  �  �  _  �  �  �  	  �  |  E  f  V  c  7  j  �  �  3    m  �  g  �    u  j  �    �  �  0  w  �  N  L  *  �    �  �  �  W  �      {  K  #  �  �  /  Z  �  �  f  @  �  �  i  �  u  x  ;  ϼ�1<#�
�49X�#�
���㼃o%   �ě���h��+���e`B�ixս�\)�ě���������\)��\)�,1���Y��#�
�\)�D���\)�\)�e`B�D���+�P�`��P�#�
���
�@�����u�D���49X�T�����P�@��D����o�u��\)�}�]/��O߽Y���`B�}󶽃o����}󶽍O߽��T������-�%�T��1�� Ž�{����ě������/��
=��`B���#������G����BtB�6B��B ēB&��A�� BE�B%�GBgB�B gkB��B#!A���BږB�|Ba�B0f9B)B��B��B6�B&SB��BRB�B:B��B!�VB,�B��B"��B%�
A�o�B��B �nB
�zB��B�bBբB*�`BB�B`=B+�B�B7�B��B�B��B�nB"
B�B �B$�.B-�B��B'ȥB BmB��B��B#�B��BwB	f�B��B͍B�UB
��B��B�.B*��B-v�BjB3ZB�mB�!B�5B ��B&��A���BB#B%A�B��B�AB GLB�RB>�A���B+B˸BD�B05B>�B�B	�B��B&@B+�B8#B=�B8�B��B" �B?�B�B";�B&?cA���B}fB �B6�B��B<KB�KB*�}B@B:�B@�B@�B�#B�BW�B<�B=�B@iB��Bq{B$=B=B�}B'�4B ?�B>�BG6B�KB�B�BB	FxB3�B=[B��B
��BŚB��B*�WB-��B��B?�A�V�@C!�A��)A��<A��BuuA�'A�J�A�A�B ��@�?�A\��@� �A��A�X�B	�PA��@A_�'A�u[=�/�AZ~6>r��A ��A>}?4O�A�&xA�o?�k�?�)AS|�A)��A,��A3�+A�g,A�?`A�yA�ӴA�k�A�
�A���Am:�A\�A��A��>A��A|jA�0%C��B�At��A�@&e�@b`8@j�WA�MEA�.@�^@@ធC�=A��%Aؔ�A�
~A�ʰA�8A�0A$S+A�2A�t�A�5�Aǆ�A�A�@��@���Aׯ�@C��A�g#Aĉ�A�u	Be�A���A���A��B �k@��A\�o@��A���AĲ)B	��A��A^�MA���>@�A[�>���A ��A??0#�A��A��m?��?�,�ASMA)�{A,��A4�A�x�A�~PA�tMA�|�A�A�cA���Al�ZAZ��A�9A�[�A�� Ax�lA�]8C���B�At�&A�V@3k+@d�@dF=A�w%A�+�@�a@�(C�&�A�PA���AЉ�A�[A���A�[�A$��A��A�	NA��A�]4AiwA�(@�:;@��   *            V         	      <       *   )   5            /      2               	      	                        0      <               !                              =      
         
      !      X   
      
                                             G                  #   7   3   )            /      /         !                  #                  '      +                                             -                           #                           !                           A                  #   !   3                     +                           #                        %                                             -                           !                           !            OS��Nd0�O��NFtxP�#GO�N/"N[�Of�O_�wO�@MO�l�P�mO��N3MN�j�N��O�zRN�IP(6O_��N��Ot��N�{�N:�bOY�N��>N<O���N�XuNe�O�ĎNV-RN�sO�QNV2�O�*O���O��Nv��N��zO��?Nb>5N*VO��OZ]uO'x�N���O5�$O2ܔNT�lP)�hNF�N���N���ND[�O
�VN�YNۮ�OF�O��N1x�N��LN�t�Os-O��SN��O�(N�A�Op�_N��N�XN�N�X    &  -    �  �  I  �  �  	�        �  2  B  �  b  �  �  +  %  �  :  �    ,  	  �  �  '    ?    �  ]  <  <  E  3  �  �  �    �  \      �    �  g  +  �  T  &  V  @  �  �  K  ^  �  g    q    |  �    �  �  �  �<49X<�t�<49X;D��%   :�o;��
;o��o��j��`B�o��C��+��t����
��9X�\)��9X���ͼě��ě���`B��h��/��h��/�����t����o�o�+�0 Ž\)�<j�\)��P��P�0 Ž0 Ž0 Ž0 Ž49X�49X�@��<j�@��L�ͽH�9�T���aG��aG��e`B�ixսu��+���-��hs���T�������㽛�㽥�T�������j��Q�����ͽ�����
=
#/=KNKHF></)04*)���

 ����������������������������#In�������U<
���EHTTValmnmmiaTHE@BEE���

�������������#&0<EC<20#��������������������������������������~�����������������~~4766:BTht|zth\B6344)6BTbkx���hB61)EHRhz��������tmaTGDEZanz����zncaZZZZZZZZ��������������������5<CHMUacbaWUPH<85555*6CNeji`OC6*
Uanz��������ztnma[UUUan�����������nUMLOU,5BN[got���tg[N75-,KOTZ[`hpha[OKKKKKKKK$0<IUbba^UKI<0##/<>D<</#-6<BKOSODB62--------<BIO[hstokjh[OJD@;7<~����������yz{~~~~~~	$%										���������������������������������-59BFNTSNB95--------�������	
	��������#0<FIIIB<90#"gmz|������zzmjefgggg����������������������������������������W^flt�����������tgYWloz�����������znghkl
")35BDGHDB=5)
������������������������������������������������������������;<HHRU^]XUQH<7;;;;;;����������������������������������������knz�������������znjk)5>BB?;53)##%,/<CGD<7/#���������������������������������������������� ������������������������������cnrz~����~zncccccccc��
#(+,# 
	�����55>BNY[aca[WNMB?7555��������������������RU^bnputtrpnhb[URLLR��������������������#)/8<=><80/,# ������������������������������������������������������������()5>BLB:5)U[gt����{tga[QUUUUUUz���������������wvzz������(*&�����������������|���������������|z||��������������������).5BQ[gt{~tmc_WNB5()������������������������������������������������������������������������������)�%���#�)�6�B�O�T�[�h�i�f�[�S�O�B�6�)��غֺѺԺֺ��������������ŠŔŒŗŠŭŴŹ����������������ŹŶŭŠ�H�G�<�<�;�<�H�U�U�^�]�U�H�H�H�H�H�H�H�H�������s�_�I�;�7�=�N�g�����������	�������ƼƳƳƳƷ���������������������������������������
����
���������������������/�%�"���"�/�3�;�?�@�;�/�/�/�/�/�/�/�/àÕÓÌÆ�z�v�{ÇÓßìó������ÿùìà�6�0�*�������*�6�C�O�\�a�c�\�O�C�6���u�j�f�d�l�x���������ûл׻Իͻ��������"��	����վ������	�"�7�<�G�J�A�;�.�"���}�_�[�������л����"����л���������������������(�5�C�N�S�W�J�5�(��H�G�C�C�F�H�P�U�Z�Z�U�N�H�H�H�H�H�H�H�H�$�������$�+�0�<�6�0�&�$�$�$�$�$�$�#������!�#�/�0�<�@�<�7�5�/�#�#�#�#�"��	��������	�"�.�;�G�N�M�J�I�D�A�.�"�������������������������������������������������}�s�{�����ƹ���
�����Ϲù���������������	������������Ϲǹù��������ùǹϹԹ׹ϹϹϹϹϹϹϹϽ����z�x�z�~�����������Ľ̽˽ý����������M�I�D�E�M�M�Z�f�f�m�j�f�d�Z�M�M�M�M�M�M������������������������������������������������������������������������N�L�N�N�S�Z�g�k�r�s�|�s�g�Z�N�N�N�N�N�N����	���'�/�'�'�����������3�2�?�C�M�Q�e�r�~�����������~�h�Y�L�@�3�ʾþ������ʾ׾��������׾ʾʾʾʾʾʽнƽĽ½Ľɽнݽ��ݽؽнннннннннǽĽ��������Ľݽ������������о������� �(�)�(����������������������������	����������������������������;�H�T�[�a�b�^�X�T�H�/�"�ììâìöù��������úùìììììììì����ŹŭŞŕŎōŔŠŭ�����������������"�����������	��"�/�;�C�G�I�N�H�;�/�"�����������������������������������������t�k�k�h�t��t�t�t�t�t�t�t�t�t�t�m�l�`�[�`�i�m�y���������������y�m�m�m�m�	����վԾ׾���	�"�2�;�G�O�G�;�1�"��	���������������������������������������������������������������������������������A�>�5�5�8�A�Z�g�s���������������~�g�Z�A�������������Ŀѿݿ�����������ԿѿĿ��<�0�#���� �#�0�<�F�I�Q�U�[�b�b�U�I�<D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�ƎƊƅƎƏƚơƳ����������������ƳƧƚƎ���|���������ĿϿѿҿڿѿпƿ�������������
�����������������������r�g�W�V�^�r�~���ֺ���������ɺ������!���������!�#�&�!�!�!�!�!�!�!�!��������!�-�:�<�:�5�.�-�!��������������������������
��
������������)�"�)�6�;�B�O�T�Y�O�B�6�)�)�)�)�)�)�)�)�����������������ûлܻ�����ܻлû��Y�Y�X�Y�Z�_�f�o�r�}�����������r�f�\�YE�E�E�E}E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��U�K�I�F�K�U�[�b�m�n�{ŅŅŁ�{�y�n�m�b�U���
����)�6�O�[�e�j�r�y�t�h�[�B�6��������������������������������������������������������������������� ��������"�"���������²©¦¡ ¦ª²¿��������������¿²²�������������}�|���������ĽԽ�ؽڽѽĽ��G�C�@�G�J�S�Y�`�l�u�p�l�`�S�G�G�G�G�G�GĚđčĊĉčĕĚĦĳĺĿ��ĿĽľĳĦĚĚĿľļĿĿ����������������������ĿĿĿĿÇ�z�o�c�U�F�<�;�<�H�K�U�n�zÓàâàÓÇ�ּϼʼ��������Ƽʼּ��������ּּ������������������������������������������#��������������@�:�5�@�H�M�Y�f�j�k�g�f�Y�M�@�@�@�@�@�@ 5 W Y Q V  r U G ) 2 . w 0 } , D D b A ` _ * = A 0 @ O q Q G / b g B D 1 = g = 9 S n 5 f m Z p 9 n X O � d  e K E : T  k H N J 0 %  . � ( y P 3  �  �  d  X  �  !  v  �  �  �  �  �  _  �  �  �  �  i  6  �    V  �  �  j  �  �  3    �  �  g  �    �  j      �  �  �  _  �  N  L  *  �    �  �  �  W  �      {  K  #    T  �  Z  �  �  f  @  �    �  �  �  x  ;  �  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  S  �  �        �  �  �  �  �  �  V    �  �    �  $  �  �    #  %  %      	  �  �  �  �  m  H    �  �  �  k  ;       *  ,  *       �  �  �  i  %  �  z    �  ;  �  �  ;  �  �  �             �  �  �  {  T  3    �  �  i    �  I  �  �  �  �  f  .  �  �  �  R    �  @  �  :    �  �   �  �  �  �  �  �  �  �  �  �  �  �  �  d  8    �  �  M    �  I  N  R  W  \  c  j  q  }  �  �  �  �  �  �  �  �  �  �  z  �  �               !  #  #  $    �  f  <  #    �  �  �  w  [  9    �  �  �  �  �  w  X  3    �  �  c  S  W  t  �  	u  	�  	�  	�  	�  	�  	�  	�  	e  	  �  M  �  9  �  �    R  c    �  �  �  �  �  �  �  �  �  �  �  S    �  �  �  N  �  o  �  �  �  �  �  �  �    
  �  �  �  �  ^    �  p  �  �  �          �  �  �  y  M     �    �  �  �  @  �  V  �  �  H  �  �  �  �  �  �  �  �  �  �  U    �  ^  �  [  �  �  �  2  ?  L  Y  ]  _  a  b  b  a  a  `  ^  ]  \  [  X  F  5  #  @  A  ?  3  %      �  �  �  �  �  n  M  (  �  �  �  �  �  l  y  �  �  �  �  �  �  s  e  T  D  3  %      �  �  �    �  �  �    <  X  a  a  U  <    �  �  t  #  �  X  �  4  �  y  �  �  p  C    �  �  �  M    �  �  �  �  }  t  �  �  �  �  �  �  �  z  d  I  .    �  �  h  %  �  �  \  �  $  C  *  !  *  +  )  &  !      	  �  �  �  �  �  K    �  ]  �  s  %      
    �  �  �  �  �  �  #  R  m  a  V  I  :  *    9  �  �  �  �  t  ^  E  (    �  �  n  -  �  �  U  �  g   �  '  /  6  :  .      �  �  �  �  x  O  #  �  �    &  �  8  �       &          �  �  �  �  �  �  �  �  w  X  #  �  �      �  �  �  �  �  �  �  �  x  ]  ?    �  �  �  V  �  ,  !    
  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  	           �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  x  S  #  �  �  W    �  �  �  M    �  �  8  �  �  �  �  �  �  �  �  �  �  �  �  M    �  s    �  W   �  '  $  !                     �   �   �   �   �   �   �   �            �  �  �  �  �  �  �  �  �    J    �  �  r  ?  ?  >  >  :  *    
  �  �  �  �  �  �  x  c  R  D  5  '      �  �  �  �  �  �  �  �  r  T  .    �  
      �  �    �  �  �  �  �  �  �  v  O  !  �  �  ?  �  X  �  �  &  �  ]  Y  H  '    �  �  �  m  C    �  �  �  Q    �  �  !  �  �    -  ;  ;  /    �  �  �  �  �  g  D  �  �  2  �  �  �  <  *        �  �  �  �  e  3  �  �  v  3  �  �  k     �  E  4  "        �  �  �  �  �  �  �  p  X  F  3    �  �  3  9  @  B  ;  3  *  !        �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  W  -  �  �  �  e  0      �   U  �  �  �  �  �  �  X  #  �  �  l    �  j  �  r  �    H   {  �  �  �  �  �  �  �  �  �            
              �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  "  �  �  �  �  �  �  �  �  �  �  �  �  w  e  X  Q  N  T  X  [  \  J  6  !    �  �  �  �  _  6    �  �  �  �  �  �  F  �          �  �  �  �  y  S  '  �  �  �  I  �  �  �    �      �  �  �  �  G  8  (       �  �  �  �  �  o  Y  T  �  �  �  �  �  �    j  X  F  5  #       �  �  �  �  �  �  q      	  �  �  �  �  �  Y  2  %    �  �  b  "  "  �  �  �  �  �  �  �  �  y  j  \  M  >  3  +  #      
     �  �  �  g  a  \  N  7    �  �  s  .  �  �    �  �  a  �        +      �  �  �  �  �  �  �  �  �  �  �  r  ]  H  =  7  0  �  �  }  o  `  Q  A  2  #    �  �  �  }  Q  '    �  �  z  T  Q  H  >  2  !    �  �  �  �  ^  /  �  �  �  -  �  Q  �  &      �  �  �  �  �  �  �  w  f  S  =  '    �  �  �  e  V  T  Q  G  =  1  &      �  �  �  �  �  �  �  �  f  +   �  @    �  �  �  �  n  J  '    �  �  �  f  K  8  -  &      x  w  y  |  �  �  �  �  �  n  N  )  �  �  �  d  6  �  w    N  ~  n  W  @  -       �  �  h  4  	  �  �  �  L  .  G  ]  *  K  C    �  �  �  �  f  F    
�  
�  
[  	�  	R    l  �     ^  U  L  B  8  (    �  �  �  �  j  7    �  �  K    �  �  �  �  m  Z  F  +    �  �  �  �  r  \  I  ,    �  �  }  6  g  `  Y  L  >  2  '      �  �  �  �  ~  `  B  $    �  �      �  �  �  Q    �    E  
  �  �  �  f  ?    �  �  �  q  c  R  ?  *    �  �  �  �  �  �  �  �  m  X  B  6    �    �  �  �  �  �  �  �  �  j  P  0    �  �  �  \  �     �  �  �    V  W  o  z  p  Z  9    �  �  i    �  U  �  s  �  �  �  �  �  �  �  �  �  �  �  �  n  T  /  �  �  �  d  +  �    �  �  �  �  �  �  T  7  =    �  �  Z    �  �  #  �  �  �  �  �  �  �  �  x  [  9    �  �  �  X  &  �  �  |  9  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  u  m  e  \  �  �  �  |  w  r  m  c  W  L  ?  3  &    
  �  �  �  �  �  y  �  }  v  l  Z  F  +    �  �  F  �  �  3  �  X  �  J  �