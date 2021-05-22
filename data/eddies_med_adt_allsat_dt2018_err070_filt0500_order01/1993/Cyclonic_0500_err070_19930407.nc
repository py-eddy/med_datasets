CDF       
      obs    Y   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?����l�     d  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�E�   max       P�a�     d     effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �Ƨ�   max       <�9X     d   t   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?B�\(��   max       @F&ffffg     �  !�   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��z�G�    max       @v}�Q�     �  /�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @-         max       @R            �  =�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�K        max       @��@         d  >\   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ���   max       <#�
     d  ?�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B/�A     d  A$   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�~M   max       B/�J     d  B�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       =[�L   max       C��*     d  C�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       =�Y�   max       C�ȯ     d  EP   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          K     d  F�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ?     d  H   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          ?     d  I|   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�E�   max       P�a�     d  J�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�Y��|��   max       ?�!�.H�     d  LD   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �Ƨ�   max       <�C�     d  M�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?B�\(��   max       @F"�\(��     �  O   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��z�G�    max       @v|z�G�     �  \�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @'         max       @Q@           �  j�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�K        max       @��@         d  k�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         =�   max         =�     d  l�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?y�_o�    max       ?�
�L/�|     �  nX                              1         J      
                           A               =                  '         
      
                     &      	         	   >   
         $               ,                                 
               -                                 O,7vO��OMiZNh�N�Na��N��,Nu�N�2�P}��Nk~N��PsܒN: N��oO��]O�׃N�W�NUQOc�4NY%�O5�N���P�cN@�XN�zOs�lOH��P�a�N^�cO��O�L�O�`N|	�PvdO{pHOn�NÑ=Nӻ�N��kO$�ZO�V�O\N$!O-�O��P<X�O�p�O�w.OG��N���N��7O��O+�Oz�Op�Py(O���OQN�N�N�KWPag#O&�N��tNQ�N�|7OCb�NN�O���NJ�INC'O9P�N`��M���O1�iO
� O/;�OsN�M�E�N]� N�ZEO)�O^,�N#*�N]��NT�N'q�N�!}Ne�<�9X<�o<#�
<o;ě�;�o;D��$�  �o�o��o��`B�#�
�#�
�49X�49X�D���T���T���T���e`B�e`B�u�u��o��o��o��o��t���t���t����㼛�㼣�
���
��1��1��9X��9X��j��j�ě��ě����ͼ�/��/��/��/��/��/��`B���+�t���P��P�����#�
�#�
�#�
�#�
�#�
�',1�,1�0 Ž0 Ž0 Ž49X�<j�@��@��@��H�9�L�ͽL�ͽL�ͽP�`�P�`�P�`�Y���%������㽡�������� ŽƧ��������
���������

�������!)+8BNY[`dd[[NB5,)! )5BJCB5)��������������������0<HUaka`UHE<00000000���

 ����������������������
#(02<?A<0)#

ox���������������{po�������������;;HT]abkaTNH;;;967;;z���������������uopz��������������������jmz|���������zmjhfjj��������������������DIbh{������{nbYVND=D��������������������Xanz����zneaXXXXXXXX*6CCHLNHC0*��������������������yz�������������~yzy��������������������|����������������yw|������������������������������������������������������������������������������������#$&�������������������~���������)6BOhy���t_O6)�#<an����vneaURF<*#_hrt���������tsmhc[_���������������������� ������ut��������������������RWVamnoz������znaTQRz�����������yxzzzzzz�����������������������������������������������������������������������tt������������tkhmtt��������������������jmsz��������|zmbbcfj/5?HN[g�������tg[B5/�����

�����������������
!
������
#0@A2<IVV]YP0,

!%0<IPVZ[WMI<30.*'$!$)/5<BBIJB55)(!"$$$$(),15>BJJLHB:52)(((()6BOcjol[OB$��������


���������bgt������������tmgcb]benr{�������{rne]]#<Un��{nUI60,*##<DKY]\UI@:0#����������������������������������������35BNOQPNBBA554333333�����<HF�����"*/11+)#	{�����������zzvz{{{{��������������������~�������������~~~~~~KNQ[bgt�������tg[NHK����������������������������������������Y[dhnpoth\[XYYYYYYYY��������������������������������������������������������������������������������LO[hrww|ztohb[POCDFLptw���������{trihknp������������������������$//,)����������������������������

�������������������������sw���������������tps %%&)4BNSSLB5) ��������������������!#&/1563/$##!!!!!!!-/<@HHJLHE<4/-------OUaahjaUUQOOOOOOOOOO�����

 �������������������������{�t�h�t�¦°²¶µ²±¦Ç��z�x�v�zÇËÓàìñòñòìâàÞÇ�������y�s�s���������������������������������������������������������������������������������!�-�2�0�-�(�!��������A�?�5�8�<�A�L�M�W�O�M�C�A�A�A�A�A�A�A�A�4�0�.�3�4�A�M�Z�]�f�h�f�d�Z�M�A�4�4�4�4�ֺҺֺֺ������ֺֺֺֺֺֺֺֺֺּ������������������ʼҼͼʼƼƼ��������������\�L�G�T�k�|�����ѿ��������ֿĿ���������������������������������������t�q�j�j�t�~āčēĕĚĠĦĦĦĚčā�t�t���r�^�N�Y�^�U�r�����ֺ��-�#��ֺ������������(�+�0�(����������������#�)�6�=�B�L�J�B�>�:�6�)����	�	����"�*�/�;�H�T�]�b�d�_�U�H�/�"�	�������������Ľݽ����� ���ݽнĽ��������������	���(�*�(�������������H�F�F�C�@�H�U�V�Z�X�U�O�H�H�H�H�H�H�H�H�	����������	�"�.�2�;�<�4�2�.�"��	���������������¿ĿϿѿӿѿĿ������������ʼż������ʼּ������������ּ��)�%�)�+�)�����)�6�B�H�G�F�F�B�6�)�)�@�4�'����$�?�M�Y�g�����������r�Y�M�@����������������������������������������ùøðìàÜØÛàêìïù����ÿÿ��ùù����������(�5�A�I�L�L�I�A�:�5�(���������w�s�x�����������ûͻƻû����������N�5����������s�������������������s�N���������������������������������������˿y�m�^�J�B�A�H�l�y�����������������z��y����������������������
�������������˾(�!����(�2�4�A�M�X�Z�\�[�Z�M�D�A�4�(��ƸƳƧƝƤƧƳ������������������������ŭŎ�o�b�m�{ŇšŨŨŹ����������������ŭ����������������$�*�0�2�5�5�3�0�$���a�T�H�?�;�5�;�K�T�a�l�z�����������z�m�a�z�q�m�t�z�������������������z�z�z�z�z�z�t�o�h�c�[�V�R�P�[�h�k�t�yāčĎĎčā�t��������������!�����������������������������������ʾ׾ؾ����׾վʾ��	����׾۾����	��"�#�.�;�>�1�.�"��	�����������������������������������������	�	�������	��"�#�+�"��	�	�	�	�	�	�	�	�(�"�����(�5�A�F�N�V�Z�[�_�Z�N�A�5�(�	� ���پҾ־Ծ׾����	�������	�׾ʾ��þ׾׾� �"�G�m�h�s�o�T�;���ݾ׽н��������������Ľн������������ݽ������}�w�s�g�Z�X�_�s�����������������������������(�4�A�M�R�U�M�G�A�4�(��ā�~āĄčęĚĦĳĶĳĳħĦĚčāāāā�#�!��
��������
��#�,�0�2�0�%�#�#�#�#���������x�l�b�[�]�e�x�����������������������������������������������������������������������������������������������̼������������4�M�Y�`�Z�M�D�'��������-�S�_�o�z�������x�_�F�2�!���û����������ûл���������������n�b�U�S�C�D�I�Y�b�n�{ŅŅŇŏŏŇŅ�{�nŠŜŔőŔŠŭűŷŭŠŠŠŠŠŠŠŠŠŠ������������������	������������������߽��`�:�.�!�� �.�S�l������������н����(���(�4�A�M�Z�f�k�s�z���s�f�Z�A�4�(�����������ÿĿſѿӿҿѿĿĿ������������������������������������������������<�2�/�(�/�5�<�H�U�^�Z�U�H�B�<�<�<�<�<�<���������������������#�)�(�%�&������������������������������������������x�x���������������Ļлڻлϻʻ»������x�L�E�@�?�@�L�Y�e�l�e�d�Y�L�L�L�L�L�L�L�L�s�g�g�s�������������s�s�s�s�s�s�s�s�s�sĦĚčā�{�vāčĚĦĳ������������ĿĳĦ¦�¦±°¦¦¦¦¦¦���������������������������������������˹ù������ùϹܹ��������������ܹϹ�¦²³¿������������������¿²¦�������������'�4�?�4�4�,�'�!����Y�M�D�M�Y�f�o�r�������������������r�Y������������������������ɺɺ��������ɺԺֺۺ���ֺɺɺɺɺɺɼ�ּټ޼���������������������¿°¨¥±²¿������������������������������ùìàÓÏÇ�Ïàù��������������àÞÓÍÓàìðóìààààààààààE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�FFFFF$F$F1F=F>F=F<F1F+F$FFFFFF�����������ù˹ùù���������������������E\ETEPECE7E4E7EBECEPE\EhEiEoEuE|EuEiE\E\���������������������������������������� 2 1 1 k ` � ! K o " E = C L b @ P T �  B p R < 9 j  / W @ r l 2 1 T , 8 + � F 8 5 < _ 2 C V B D 6 U B 5 F Z R H 5 0 D 2 N V 7 f J I k b B 8 [ [ o ! D 1 g 4 v A 4 M ` _ S [ P P    {  -  �  �  �  �    I  &  �  z  #  #       "  �  �     �  `  �    �  Z  S  �  �  J  w  �  �  ,        �  �  1  �  m    1  ?  r  n  �    �  �  �  �  �  ?  s  
  �  �  �  5  �    �  �  S  �  �  Y  �  \  S  �  �  2  t  R  u  0    �    s  �  +  �  �  J  �  L�o<#�
�49X;D���o��o��o�o�#�
�L�ͼt���t���1��o��1�����'�C���1����o��㼬1���
������,1��P���w��j��w��w�\)�����q����������`B�o�t��]/�#�
��h�49X�L�ͽ���m�h�t��]/�t���w�\�<j�H�9�T������e`B�aG��,1�49X����Y��]/��7L�e`B��%�@�����L�ͽL�ͽ�o�ixսH�9��hs�}󶽗�P����Y��ixս�%��C����-�����9X��E�������ͽ��B�zB��B�!B�B+�*BO}BA�B�B%2B*��B HA���BBB+�A���BrzB)�B:"B
�B/�AB:fB3�BB ��B3WBB�Bf B ��Bg�B
� B�BBԖBhvBQ`B�xB�-B�B
�B+SB��B ��B�&B�"B!�A��6B	�B�0B"�4B%�B&N�B1�By�B%�BuB
R�B)�B&�dB%�oBF�B�B�aB�dB~JB�PB�yB
�LB	�gB�-B�@B��B��BM%B*�BC�BtsB
�B��B�B"��B#ԖB-��BM~BC�B.�B+�B��B 9B��B!�B�)B<fB?7B@B+YfB`{B<�B<�B$�B*\eB8�A�~MB�BB=%A���BÈB)AB�B��B/�JB�B=9B?eB �sB6CBSOB�`B �*B�B
�.B?�B>RB��BG�B�B�DB1�B
�	B�.BF�B ��B�?B�B!��A���B��B��B"?B&PB&f!B;MBBB@	B8.B
='B)6YB'�DB%DfB?DB�gB�]B��B��B�RB��B
��B	�jB��B �B��BnzBrCB?*B�@BAB
#�B�B��B"� B$�B-�B@B�dBAVB9�B�B'�B�{B>�A�oA��Aq�]A�@d_PA:�A<h�@Bп@�XAt�cAH�^A�(�@->rA5gTA֭1A��A)/A�.A�(�A\�"Aw�	A�A�X�@�qiA� TA̿�A�A|@���A��A�yAm��A���A:"QBU�A��kB�oA�<�A�~A�H�A��APH�A[��A!�eA�I;A��)AX�IAXݣA(��A�A6��A�xrA�s�@��8A�`A�9@�أ@�,�@��RA�KA���A��A��A=>�Aw��B��A�"A�CA��X@���?�z_A�(A�V�A�s!A���>�yA�� @�2N@�D�@T�W@9�JAG�A�h+A�%A˵C�%C��*=[�LC��IA���A��iAɀAqA��`@c�A:�}A:�@D5@���Aw~}AG��A��)@;�BA5A�|�A�|BA)*A��AAĀ�A[}Ay�Ap�A�~�@�P�A�/�Ă'A�ƛ@���A��A�x�Ak#yA��A:�MB:�A���B	s�A�x�A��
Aڇ�AӀAOҴA\ �A!�A��A��HAY]AW�A(�pA�wtA8�AߊSA�;@��tA�zA䂹@�f�@u��@�hA�*�A�X~A�CA �A?��Aw]iB>�AÀ�A�zA��@�Ԥ?�JA��KA�y�A�|�A�~i>��jA�w=@�{�@�v�@U�b@4)�A��A�~TA�eAˑ�C�'>C�ȯ=�Y�C��}A��               	         	   	   1         K                  	               B               =                  (         
      
                     '       
      	   
   ?            $               ,                                                -                  	                                             9         ;            #                     -               ?      )   !         3                                    =   '   %                        +               =                     !                                                                                          '         )                                 )               ?      '                                                                              #               9                                                                                 N�M�O��O��Nh�N&ʤNa��N��,Nu�N�2�O���Nk~N��P��N: N��oOb�N�m1N�W�N��OPh^NY%�O5�N���O�"N@�XN���N���O-f�P�a�N^�cO��2O��KN�"�N|	�O�8O{pHO"?�N��fNӻ�N��kO$�ZO?@N�V�N$!Oa:O0ȜO�@]O�F�O���O5<�N���N��7O���O+�N��wOiZO���O���OQN�N�N�KWPX�iO&�N��tNQ�NI2(OD�NN�O{��NJ�INC'O9P�N`��M���O'?�O
� O/;�OsN�M�E�N]� N�ZEO�O^,�N#*�N#'N,AQN'q�N�!}Ne�    �  J  h  V  �  m  �  �  �  �  �  N  s  �  �  �  '  S    �  �  0  5  �  �  �  �  2  �  b  �    �  �  h  `  I    �    �  �      i  �  g    �  n  C  		        >  �  �  �    #  �  �  b  �  �  �  c  b  $  �  �  �  C  �  I  
u  �  B  _  &  �  m  �  i  7  @  $<�C�<�o;��
<o;D��;�o;D��$�  �o��1��o��`B���#�
�49X�D�����ͼT���u�e`B�e`B�e`B�u��9X��o���
��/��t���t���t����
���
���
���
�+��1���ͼě���9X��j��j���������ͼ�`B�C���w����`B��h��`B�����t��#�
�#�
�49X���#�
�#�
�#�
�'#�
�',1�49X�@��0 Ž49X�49X�<j�@��@��@��L�ͽL�ͽL�ͽL�ͽP�`�P�`�P�`�]/��%������w���
������ ŽƧ�����������������

�������35@BINT[\`^[NB950+33 )5BJCB5)��������������������0<HUaka`UHE<00000000���

 ����������������������
#(02<?A<0)#

z{����������������~z�������������;;HT]abkaTNH;;;967;;y{���������������~y��������������������jmz|���������zmjhfjj��������������������kno{���������{{qnmkk��������������������`anz����znha````````*6BGJKJC6+)��������������������yz�������������~yzy��������������������|�����������������}|������������������������������������������������������������������������������������#$&�������������������~��������)6B[ev���tOB6/)#<Banz����tnaUG<,#aht���������ttnhf]aa�����������������������������������������������������UV[_anqwz�����zna]UU}���������|z}}}}}}}}������������������������������������������������������������������
������otv����������tokoooo��������������������kmtz��������zwmfddfk@BBFN[gtv����tg[YN@������� ����������������

���������
#0<DIPQY[WM<0#
(0:<IOUYZVI<50.+(%#($)/5<BBIJB55)(!"$$$$(),15>BJJLHB:52)((((")16BOUahlh[OB/��������


���������nt���������vtjnnnnnnjqz{�����������{vpnj#<IUp}{nbUI:20*'#<DKY]\UI@:0#����������������������������������������35BNOQPNBBA554333333�����;GE�����"*/11+)#	{�����������zzvz{{{{����������������������������������������V[gjt��������tg[WOVV����������������������������������������Y[dhnpoth\[XYYYYYYYY��������������������������������������������������������������������������������LO[hqvu{ytlhd[QODEFLptw���������{trihknp������������������������$//,)����������������������������

�������������������������tx��������������|trt %%&)4BNSSLB5) ��������������������"#.//452/#" """"""""./<?GHIH<5/-........OUaahjaUUQOOOOOOOOOO�����

 �������������������������t¥¦¦²´³²®¦Ç��z�x�v�zÇËÓàìñòñòìâàÞÇ�����y�x�y�z�������������������������������������������������������������������������� �!�#�)�!�!����������A�?�5�8�<�A�L�M�W�O�M�C�A�A�A�A�A�A�A�A�4�0�.�3�4�A�M�Z�]�f�h�f�d�Z�M�A�4�4�4�4�ֺҺֺֺ������ֺֺֺֺֺֺֺֺֺּ������������������ʼҼͼʼƼƼ����������������y�m�d�d�m�y�������Ŀݿ���ؿĿ���������������������������������������t�q�j�j�t�~āčēĕĚĠĦĦĦĚčā�t�t�������~�w�p�n�|�������ֺ���	��� �ֺ��������(�+�0�(����������������#�)�6�=�B�L�J�B�>�:�6�)���� ���"�+�/�8�;�H�T�\�a�c�]�T�Q�H�;�/� �Ľ����������ýĽнݽ�����ݽԽнĽĿ��������	���(�*�(�������������H�H�H�E�B�H�T�U�X�W�U�I�H�H�H�H�H�H�H�H�	���������	��"�.�6�;�;�3�1�.�"��	���������������¿ĿϿѿӿѿĿ������������ʼż������ʼּ������������ּ��)�%�)�+�)�����)�6�B�H�G�F�F�B�6�)�)�M�@�4�&�!�$�(�@�Y�f�~�����������r�g�Y�M����������������������������������������ìààÛÞàìù��ÿûùìììììììì���	����$�(�5�7�A�B�A�>�5�5�(����������y�x�����������ûƻû»������������N�5����������s�������������������s�N���������������������������������������˿m�c�O�F�I�]�m�y�����������������}�v�y�m��������������������������	�����������˾(�"�� �(�4�7�A�M�V�Z�Z�Z�Z�M�B�A�4�(�(��ƸƳƧƝƤƧƳ������������������������ŭŠŉŇŕŠŦŭŹž����������������Źŭ����������������$�*�0�2�5�5�3�0�$���h�a�T�H�F�D�H�T�W�_�m�z���������}�z�n�h�z�t�r�y�z���������������z�z�z�z�z�z�z�z�t�o�h�c�[�V�R�P�[�h�k�t�yāčĎĎčā�t��������������!�����������������������������������ʾ׾ؾ����׾վʾ����	����������	���"�*�.�5�5�.�"������������������������������������������	�	�������	��"�#�+�"��	�	�	�	�	�	�	�	�(�$�����(�5�A�C�N�T�Z�Z�]�Z�N�A�5�(�	��������������	�������	�׾Ͼʾɾξܾ����	��"�(�.�5�.�	����׽Ľ����������Ľнݽ�����������ݽн����������z�y�s�g�s�������������������������������(�4�A�Q�T�M�E�A�4�(���ā�~āĄčęĚĦĳĶĳĳħĦĚčāāāā�#�!��
��������
��#�,�0�2�0�%�#�#�#�#�x�l�e�_�]�_�g�l�x���������������������x�����������������������������������������������������������������������������̼'�������%�'�4�@�M�S�Y�[�T�M�@�6�'������!�-�S�c�o�v�y���x�_�F�:�-�!��û����������ûл���������������n�b�U�S�C�D�I�Y�b�n�{ŅŅŇŏŏŇŅ�{�nŠŜŔőŔŠŭűŷŭŠŠŠŠŠŠŠŠŠŠ������������������	������������������߽��`�:�.�!��!�.�S�l�����������н����(���(�4�A�M�Z�f�k�s�z���s�f�Z�A�4�(�����������ÿĿſѿӿҿѿĿĿ������������������������������������������������<�9�/�+�/�:�<�H�U�W�W�U�H�>�<�<�<�<�<�<�� ������������������$� ��������������������������������������������껅�{�������������������»λͻɻ»��������L�E�@�?�@�L�Y�e�l�e�d�Y�L�L�L�L�L�L�L�L�s�g�g�s�������������s�s�s�s�s�s�s�s�s�sĦĚčā�{�vāčĚĦĳ������������ĿĳĦ¦�¦±°¦¦¦¦¦¦���������������������������������������˹ù������ùϹܹ��������������ܹϹ�¦²³¿������������������¿²¦�������������'�4�?�4�4�,�'�!����Y�M�D�M�Y�f�o�r�������������������r�Y������������������������ɺɺ��������ɺԺֺۺ���ֺɺɺɺɺɺɼ�ּټ޼���������������������¿²©§²´¿������������������������������ùìàÓÏÇ�Ïàù��������������àÞÓÍÓàìðóìààààààààààE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�FFFFF$F-F1F5F:F1F*F$FFFFFFFF�����������ù˹ùù���������������������E\ETEPECE7E4E7EBECEPE\EhEiEoEuE|EuEiE\E\���������������������������������������� + 1 ) k W � ! K o  E = > L b 3 6 T �  B p R < 9 = # * W @ p n 3 1 ' , @  � F 8 7 : _ - 7 % ; F 6 U B , F P 4 B 5 0 D 2 N V 7 f F D k c B 8 [ [ o ! D 1 g 4 v A 0 M ` \ M [ P P      -    �  j  �    I  &  (  z  #  �       �    �  �  �  `  �    C  Z  �  �  v  J  w  �  �          p  �  1  �  m  �  �  ?  V  �  �  ~  Z  |  �  �  ;  ?  �  <  �  �  �  5  �    �  �  S  d  ;  Y  d  \  S  �  �  2  c  R  u  0    �    X  �  +  L  X  J  �  L  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  �  �  	        �  �  �  �  X  *  �  �  \  �    <  l  �  �  �  |  t  y  �  �  �  �  �  n  L  )    1  I  [  P  D  9  �  *  <  G  J  I  D  :  ,    �  �  �  �  V  )  �  �  �  �  h  X  H  7  '      �  �  �  �  �  �  }  k  Y    �  �    9  ?  F  K  O  R  T  V  T  R  N  I  C  ;  2  *       �  �  �  �  �  �  �  �  �  �  t  f  X  J  >  3  (            m  c  X  N  B  7  *      �  �  �  �  �  �  �  �  q  A    �  �  �  �  �  v  c  O  7    �  �  w  ?    �  �  S     �  �  �  �  z  c  L  8  #          �  �  �  �  �  U     �  �  �  	  /  Q  r  �  �  �  �  �  T    �  `  �  x  �  �  !  �  �  �  �  �  �  x  f  S  =  '    �  �  �  �  �  �  k  R  �  �  �  u  e  U  I  =  2  (         	      �  �  �  �  J  �  �  +  H  N  J  <    �  �    �  !  �  �  i  �    C  s  q  p  o  i  b  Z  C  #    �  �  �  s  K  $  �  �  �    �  �  �  �  �  y  c  F    �  �  _    �  �  �  w  e  W  H  [  y  |  r  c  P  =  &    �  �  �  �  �  n  L    �  �  w  1  G  N  D  .    D  c  {  ~  u  b  G  )    �  �    4    '  #            �  �  �  �  �  �  �  �  z  d  N  8  #  :  ?  C  I  O  U  ^  g  a  Z  Q  G  2  �  �  W  8    �  �          �  �  �  �  �  �  �  �  w  a  C     �  �  J   �  �  �  �  �  �  z  r  j  b  [  V  S  Q  O  L  J  H  F  C  A  �  v  ^  A    �  �  �  e  -  �  �  �  V    �  J  �  �  D  0  .  ,  *  '  $  !        �  �  �  �  �  v  `  <     �    *  5  2  +    �  �  u  :      &    �  �  K  �  Y  j  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  t  R  )  �  �  �  b  
  �    v  �    @  d    �  �  �  �  �  �  �  b  0  �  �  h    �  �  �  �  �  �  �  �  �  �  t  W  6    �  �  �  e  D  �  �  <  2  &    �  �  �  �  �  l  b  L  B  0    �  [  �  Y  �  �  �  �  �  �  �  �  y  m  `  T  G  ;  '    �  �  �  �  �  l  C  Y  b  `  _  ^  V  J  ;  $    �  �  �  �  �  V    �  $  �  �  �  �  �  t  E    �  �  �  s  /  �  �  P  �  �  Q        �  �  �  �  �  �  �  �  m  I  !  �  �  �  V    �  J  �  �  �  �  �  �  �  �  �  �  |  q  g  \  Q  G  <  /  #    �  �  A  �  �  �  �  �  �  �  �  n  <    �  ~  $  �     �  h  Z  L  <  )    �  �  �  x  S  4    �  �  �  {  Y  ;    #  A  O  Y  ^  ]  S  B  /    �  �  �  �  �  �  �  �  Z  "  :  9  8  ?  G  B  :  /  $    
  �  �  �  �  �  �  �  m  E         �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  g  M  �  �  �  �  �  �  �  q  Z  <    �  �  �  T    �  �  b  )          	    �  �  �  �  �  �  �  �  �  �  {  o  `  M  �  �  �  �  �  �  �  �  �  {  W  +  �  �  {  "  �  "  �  �  �  �  �  �  �  �  �  |  s  h  Y  D  %  �  �  �  �  o  [  E    
    �  �  �  �  �  �  �  �  �  �  |  l  X  C  -                �  �  �  �  �  �    V  !  �  �  >  �  �  ;  �  #  <  N  Z  f  i  b  P  3    �  �  �  Y    �  ?  �  �  �  �  �    Q  u  �  �  w  a  C  "  �  �  �  5  �  T  �   �    9  `  f  a  W  E  4  "  �  �  �  h  G  1    �  v  �   �  �  �            �  �  �  �  z  M  !   �   �   �   �   �   �  �  �  �  �  �  �  �  �  `  8    �  �  i  '  �  �  (  �  �  n  h  a  [  V  N  ?  1  !       �  �  �  s  +  �  �     �  C  1       �  �  �  �  �  �  k  O  3    �  �  �  �  �  (  �  	  	  �  �  �  �  ]  3    �  �  �  ?  �  e  �  �  �  v      �  �  �  �  �  �  �  �  �  |  e  Q  A  3    �  �  t  �  �  �              �  �  �    R  )  �  �  P   �   �  �  �  �          �  �  �  �  �  �  �  a  A  "    �  E  �    /  <  8  *    �  �  �  �  ^  Q  8  �  �  l    �  l  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  y  j  V  ?  (    �  �  �  m  9    �  �  f  R  V  S  �  �  �  �  }  u  n  f  ^  W  T  W  Z  ]  a  d  g  j  m  p          �  �  �  �  �  �  �  q  R  2    �  �  �  �  {     !    �  �  �  �  _  )    �  �  �  |  A    �  �    W  �  �  �  �  �  u  X  ?  ,      �  �  �  �  �  u  S  0  
  �  �  �    i  Q  6    �  �  �  �  �  y  c  N  :  !  �  �  b  +  �  	�  
V  
m  
}  
�  
�  
�  
�  
�  
�  
�  
�  
�  
�  
�  
�    �  �  �  �  �  �  d  8  
  �  �  u  =    �  �  R    �  �  �  �  �  �  �  �  �  ~  i  L  "  �  �  �  P    �  %  ]   �  �  �  r  ]  I  1    �  �  �  �  i  E  "   �   �   �   �   e   ?  B  b  `  V  A  (    �  �  �  f  8    �  z    �  {  %  t  b  W  K  @  1  !       �  �  �  �  �  w  K    �  �  �  L  $      �  �  �  �  �  �  �  n  Z  F  3    '  8  I  Z  j  �  �  �  d  A    �  �  �  q  P  J  E  B  ;  /  "    �  I  �  z  X  +  �  �  �  F    �  �  �  �  k  8    �  �  [  !  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  y  @  C  =  4  &      �  �  �  �  I    �  :  �  �  �  �  r  �  z  o  ^  L  ;  .  &         	  �  �  �  �  �  �  R  �  I  .    �  �  �  �  �  �  r  [  ?    �  �  �  [    �  p  
u  
p  
a  
E  
  	�  	�  	�  	U  	  �  �    �    �  f  �  s  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �      B  =  7  2  -  (  "      �  �  �  �  �  �  �  q  O  .    _  W  N  F  @  ;  3       �  �  �  �  s  P     �  �  P   �  $  &         $  !        �  �  U  �  g    �  �  S  &  �  �  �  �  �  �  �  �  �  �  j  L  ,  
  �  �  �  Y  "  �  m  ^  N  =  *      �  �  �  �  �  �  �  �  �  �  i  &  �  �  �  �  �  �  �  �  �  �  �  s  \  =    �  �  �  y  P  '  T  _  g  W  E  /       �  �  �  �  �  h  -  �  �  Z    �  7  &      �  �  �  �  �  �  �  �  y  m  `  ]  ]  ]  ]  ]  @  >  3    �  �  �  m  ;    �  �  r  J  $  �  �  �  *  �  $    �  �  �  �  �  o  T  9    �  �  �  v  F    �  �  �