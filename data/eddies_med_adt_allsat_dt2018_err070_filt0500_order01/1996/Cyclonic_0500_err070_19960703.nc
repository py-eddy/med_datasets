CDF       
      obs    G   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�333333       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��   max       P�4�       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ����   max       <�9X       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��Q�   max       @F\(��       !    effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �ҏ\(��    max       @vpQ��       ,   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @/         max       @Q@           �  70   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�L        max       @��@           7�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �
=q   max       ��o       8�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�\'   max       B4��       9�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��;   max       B4��       ;   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >I+r   max       C�b�       <0   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >@`�   max       C�vj       =L   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          L       >h   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ;       ?�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          1       @�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��   max       P(��       A�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�����A   max       ?�u��!�/       B�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ����   max       <��
       C�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��Q�   max       @F\(��       E   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�G�     max       @vpQ��       P(   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @%         max       @Q@           �  [@   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�L        max       @�,�           [�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         BY   max         BY       \�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?{�Q�`   max       ?�tS��Mj     0  ^   +            
            =                      
                  G   @   /         7   	                  #   	      >         K   	         A                                     	               !   %   !                  G   	   P��N� �N�N�q�N���N>h�N�`dO8�APV�N� �OvO���N�$�O@�>O0p.OVN=�&OicO{�O[OG�O�p�O���P��N{�N�!�O���O+54N;9vP(��O)C�M�apO���PYN�5�O�9�P+��N��aN%�$P�4�N�g�Oc� O��8P6�lN�	�Oy7�O��Of�N�.�M�\�N�C�NU�OL�eO��O�j�O:T�N�</N�1�O�ۨN�w�O�p�P$�O:Z�N�`vN���M��O��O+.aO��|N���O7n�<�9X;D��;o:�o��o��o���
���
�t��#�
�49X�D���D���D���D���e`B�e`B�e`B�e`B�e`B�u�u�u��o��C���C����㼛�㼣�
��1��1��9X��9X��j��j��������������/��h�������o�+�+�C��C��t��t���P��������w�#�
�#�
�0 Ž0 Ž8Q�@��@��T���T���Y��]/�u��%�����C������
#/KU[^`]U/#
������$����������������������������W[hhjt���{tiha[WWWWW��������������������NOS[__][OJKKNNNNNNNN��������������������#/<HKJLH?<7/)#=E[gnt�������n[NB<9=����������������������� 	"&"	��������
/8=A;3#
�������v����������}vvvvvvvv��������������������*+6=BEC>6*��QTYamz}zslkga^TPLJKQmmwz������zmmmmmmmmm����������������)5BN[gikd]TNB5)y��������������~{yyuz{������������zvttu��������������������������������Taz����������zaTLKOT=BOT[bb[OBA9========fhot}�����thhbffffff<MYanuz}vvxnaUH@966<tt��������������tnjt#,/:<B</# #$#%05<@NR^������{U<0'%��������������������OOT[_b[OOOPOMOOOOOOO�����
#++%
�������\fkpz}��������zga[Y\�� 

�����������������������������������������������GIMUbnnpnnfbUKIIGGGG��������������������%1;U[g���������tgB*%DHLTTVaabcba\THFCBDDamz������zrmea]ZZY[aABO[hv��{rnph[O@;=?A������%#������������� �����������������������������~~�t����������������rot')012+)���������������������)6B86)"��������������������&)35BGMFB5,)&&&&&&&&R[bgt����������tg[OR#,0<IIQSMI<0#&6BO[iopnh^[OB2,)& & )5<BA<;@EB5)"IORZ[^hotutkh[OEIIII��������������������HPU\bjvxyvtnb^XNECCHQ[gt����tg[VQQQQQQQQ����"!!��������%'6DFB6)�������������������������������������������������	�����"#%%##��������������������:<>FHUajnstpnb_UHC9:���#/>HJH/#
��������tsstx}�����������ntz�����������zvnmlnőŐŢŤŭŲŹ��������� ����������ŭŠő��ںֺκѺֺ����������������⾱�����������������������ľ¾������������Ŀ������������Ŀʿѿֿݿ޿ݿܿѿĿĿĿľ������������������������žľ�������������������	���	�������������(�'���(�)�5�A�H�N�S�N�J�A�;�5�(�(�(�(�U�H�=�>�H�T�U�a�n�r�zÂÇÍÇÅ�z�n�a�U�)��	����ùÖìù��������6�P�W�V�O�B�)�������������)�6�;�6�4�)������"���	�� ��	���"�/�5�;�B�J�H�;�/�"�)�%����5�N�[�t�t�g�[�N�)�4�/�(�*�4�A�M�X�N�M�B�A�4�4�4�4�4�4�4�4�`�_�T�Q�L�I�K�T�`�m�{�����������~�y�m�`�;�1�.�"�����"�.�<�G�P�R�P�X�^�T�G�;�t�m�h�X�O�X�[�]�hāčĚĥĦĮĦĚčā�t�B�B�6�2�0�6�B�O�U�O�I�N�B�B�B�B�B�B�B�B�5�2�(�(�5�A�N�`�g�s�u�����~�s�g�Z�N�A�5�g�c�Z�N�C�=�<�?�A�N�Z�s�����������t�s�g����ݽĽ����������Ľнݽ����������Ç�{�z�q�n�j�k�n�zÇÓàâñõôìàÓÇ�����������������&�3�8�8�5�0�$�������Ěčā�t�k�t�zāčĦĳĿ������������ĳĚƤƔƈƉƎƒƧƳƾ������������������ƳƤ�U�P�Q�U�\�a�n�t�v�o�n�a�U�U�U�U�U�U�U�U�<�:�/�*�/�6�<�H�U�\�^�U�T�H�<�<�<�<�<�<������������������������ ����ܹù������������������ʼּܼ޼ۼ�ݼڼּܼʼ��нνнѽݽ߽������ݽ׽ннннн����������s�Z�N�A�#�,�g����������������������ݿԿѿ̿ȿɿѿݿ������������L�L�Y�]�e�r�y�r�r�p�e�b�Y�L�L�L�L�L�L�L���������o�f�c�p���������������������������������������	�"�/�H�P�[�]�Y�H�;�"�	����������������� ����&�������������������׾ʾ������������׾�����	�����x�_�F�S�^���������л�������ܻû������A�:�4�-�)�4�4�A�M�M�Z�_�b�Z�N�M�A�A�A�A�a�`�a�f�n�o�zÇÊÇ�|�z�p�n�a�a�a�a�a�a�/���²¦¿������H�P�R�X�^�H�/�s�k�g�Z�Z�Z�_�g�s�������������������s�s�����޿������*�5�<�A�P�A�5�(�����x�s�h�l�s�x�����������ûԻۻջû������x�Y�Y�m�r�~���ּ���'�'�����ּ����r�Y�����������������������������������������ֺκɺźú��ɺֺ��� ��������������������������������� �!�������Z�Y�P�M�F�M�N�Z�f�s���������v�s�f�Z�ZŠŞřŔŗŠŠšŭŮŰŶŹŻŹŭŠŠŠŠ�ܹܹع۹ܹ�����ܹܹܹܹܹܹܹܹܹܹܽ����������������ĽŽнӽݽ��ݽнĽ������������������������������������������������z�w�t�r�u�z����������������������������������������������������������������s�^�h�s�w�����������������������������sŠŔŐŎŔŠŭŹ������������������ŹŭŠ�ù��������������¹ùùϹӹٹٹϹùùù�ŹůŭŤŭŹ��������������������ŹŹŹŹ�û��������������л���� �����ܻлþ�����������	����
�	�����������������������������������ʼ�����׼мμʼ����:��������.�G�������ƽҽý����y�`�:ā�t�h�f�b�h�rāĚĦĳľĿ��ĿĳĦĚĎā�_�X�U�]�_�l�x����������x�l�_�_�_�_�_�_�`�^�T�H�G�;�6�;�G�O�T�`�m�w�y�u�m�b�`�`E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��
�����������������
��#�+�1�0�-�#���
���������������������ĿѿԿݿݿܿԿѿĿ�D�D�D�D�D�EEE*E<EPEWEhEgEUEPE?EED�D��\�\�\�R�O�C�A�6�*�%�*�.�6�C�O�\�\�\�\�\�����������������������ʾӾ޾۾׾̾ʾ��� B @ G J ; Q %   f e - ^ + i w L > I k - " / 7 H , > � Z Z C � Z 7 J P 6  � N V f 8 b A 5 4   h s = D ] W @ S : B X j U w R   6 � (   i 0 4    m  �  �  �  �  m  �  �  �  �  �    �  �  �  �  `  �  7  �  �    �  ?  �  �  �  �  k  -  {  G  �  z  �  `    �  �  �    7  J    �          Y  "  {  �  W  M  �  �  �  �  �  �  �  �  �    ,  �  k  �  �  ����
���
���
��o�#�
�ě��u��9X��O߼�t���o�49X�u�ě���C���j��t���`B�����+��㽰 Ž�����%���ě����P��/�+�0 ŽC���`B�@��m�h�o�0 Ž�-��P�o������w�T���y�#�ě���P�P�`�D���8Q��w�#�
�D���']/�aG���\)�H�9�m�h�H�9�m�h�P�`���
���罬1�����%�}󶽟�w��vɾ
=q���-��B�iBM�B!e�B�QB4��B�7B��B!B	�WBKWA�\'BT�B3�B+^�B/e�A��A��pBB�RB)�VB �VB�B�A���B��BN�B�-B��B3�B'�B*M�B
Bl�B O�B��B^�B��B'�EB"�|B	p�A�y_A�%�B5B-I�B��B�,B<vB��B+BdwB!��B[�B
�B%�B�-B �BN�B&AB'�B	U~B�B|�B�zB!#]B:�B�oB�B��B�B
nPB�WB+B�B!EfBéB4��B�B��B�B
2�B@�A��;BPBv�B+"�B/��A�h�B :ABB��B)�B ��BףB�8A�tBB�B@+Bm�B B?�B(�yB*A�B?�B?�B A�B� B��B�FB'��B#6,B	>A���A���B?�B-0NB�GBäB;�B��B��BA[B!�,BTB
?�B&?:B��B�zBJ�B8B'�B	��B<�B��B�B!4�B@7B?�B>�B<'B�IB
BRB��A��x@FL�AK?nAy�|AK߁AXg�A���A�mCA��A�%�A���A�q�A:8#Aj��Ab��A�i+A؁tA�`�A��A)��A�8pB��A��B�bAƣ�A�W�>ȫ�A 4(A+L�A�A��?���A��kA��KA�6�AR��@���A;��A��dA�eQA�NA��m@�ICAD"A�#s@H5�A�3AA*�A�-�>��A&�^A���A�h�A ��A�a.A���>I+rA�q_@���AY^|@��A�A�L#@���Ah�`C�b�A��Aw��C��B �KAM��A���@D *AL�5AyC�AL��AX,{A��2A��A�KA�|A��>A���A:\bAj�qA`��A���A�p�A��A�p�A*ҲAɂyB��A���B�0AƀAĄ&>ׇ+@�/�A+fA�k�A�?���A�*
A��Aҍ�AR�@��AA;&
AȅA��HA�XA���@��%A|A���@G��A�kmA@�IA�Vz>�\A&��A�3VA�oA �zA���A���>@`�A�w�@�>�AY(~@��A:9A�v�@�BAi��C�vjA�}�Ax�C���B ��AM�~   +   	         
            =   	      !            
                  H   A   0         8   	                  $   
      >         L   	         A                                     
               "   %   !         	         G   
      #                        1         %                              !      %         !         1         #   '         /         ;            ;                                             !         5                     '         #                                                                                       1            !         '         '            '                                             !         #                           O���NK@�N�N�q�N���N>h�N���N�AN�zcN� �OvOenrN�$�O+)[O0p.OVN=�&OP��O{�N�
oOG�O��]OQ{�O�G�N=�N�!�O�OQ�N;9vP(��O)C�M�apO��4O��N�5�O�9�O��cN��aN%�$P�|N�g�OA'iO��8O���N�	�OfКOl�'Of�N�.�M�\�N��NU�OL�eN�F�O�j�O:T�N)�pN�1�O�ۨN�w�Op
�O�?�O�"N�`vN�U�M��O[ΦO+.aO�_"N���O7n�  �  �    �  O  �    �  	�  S  �    �  �  �  S  �  g  �  �  �  	�  
T  %  �  �  	,  �  K  X  >  1  �  u  �  Q  �  �  W  p  �  �  Q  �  �  �  I  �  �      #  �  �  �    �  
  �  !      �  �  w  �  �    }  �  �<��
:�o;o:�o��o��o�ě��t��<j�#�
�49X���ͼD���T���D���e`B�e`B�u�e`B��o�u���+��`B���㼋C��0 ż��
���
��1��1��9X��`B��h��j������P������/�ixռ��+���D���+�C��\)�C��t��t��������#�
��w�#�
�<j�0 Ž0 Ž8Q�P�`�ixսe`B�T���]/�]/�y�#��%���罋C������
#/HUZ]^[U/#
�����"��������������������W[hhjt���{tiha[WWWWW��������������������NOS[__][OJKKNNNNNNNN��������������������#/5<DB@<6/#!fgkptw��������vtg`ff����������������������� 	"&"	�������
#+/3421(#���v����������}vvvvvvvv��������������������*+6=BEC>6*��QTYamz}zslkga^TPLJKQmmwz������zmmmmmmmmm�����������������)5BN[gikd]TNB5)��������������������uz{������������zvttu��������������������������������TYamz�������zmaWQPQT@BOO[\`[OCB;@@@@@@@@fhot}�����thhbffffff;<>HQU[aeffeaaUHA<<;lt}��������������trl#,/:<B</# #$#%05<@NR^������{U<0'%��������������������OOT[_b[OOOPOMOOOOOOO�����
#&(&#
������`chz�����������zph``�� 

�����������������������������������������������GIMUbnnpnnfbUKIIGGGG��������������������BMgt���������[N@?A@BDHLTTVaabcba\THFCBDD\acmtz�����~zma]\[Z\ABO[hv��{rnph[O@;=?A������������������� �������������������������������pv���������������utp')012+)���������������������)6B86)"��������������������&)35BGMFB5,)&&&&&&&&R[bgt����������tg[OR#/0<EIOQKI<0#&6BO[iopnh^[OB2,)& & )5<BA<;@EB5)"NOY[hhohc[ONNNNNNNNN��������������������HPU\bjvxyvtnb^XNECCHQ[gt����tg[VQQQQQQQQ�������������� $-/) ���������������������������������������������������������"#%%##��������������������:<>FHUajnstpnb_UHC9:���
#&-/22#
��������tsstx}�����������ntz�����������zvnmlnœőţťŭųŹ��������������������ŭŠœ���ֺҺԺֺ���������������⾱�����������������������ľ¾������������Ŀ������������Ŀʿѿֿݿ޿ݿܿѿĿĿĿľ������������������������žľ�������������������	���	�������������5�0�(���(�+�5�A�F�N�Q�N�I�A�7�5�5�5�5�U�U�H�A�C�H�M�U�a�n�r�zÁ�~�z�n�a�U�U�U������������������������%����������������)�6�;�6�4�)������"���	�� ��	���"�/�5�;�B�J�H�;�/�"�g�[�N�B�6�,�-�5�=�N�[�g�t�x�t�g�4�/�(�*�4�A�M�X�N�M�B�A�4�4�4�4�4�4�4�4�m�`�T�R�M�K�N�T�`�m�v�������������|�y�m�;�1�.�"�����"�.�<�G�P�R�P�X�^�T�G�;�t�m�h�X�O�X�[�]�hāčĚĥĦĮĦĚčā�t�B�B�6�2�0�6�B�O�U�O�I�N�B�B�B�B�B�B�B�B�4�*�5�A�G�N�Z�d�g�s�����}�s�q�g�Z�N�A�4�g�c�Z�N�C�=�<�?�A�N�Z�s�����������t�s�g�ݽ׽нĽ����������ĽȽнݽ�������ݽ�Ç�{�z�q�n�j�k�n�zÇÓàâñõôìàÓÇ������������������$�+�/�0�)�$������ĳĦĚčĂā�ąčĕĦĳĿ����������ĿĳƱƧƛƕƓƖƚƧƳ��������������������Ʊ�U�S�T�U�a�a�n�r�u�n�m�a�U�U�U�U�U�U�U�U�<�:�/�*�/�6�<�H�U�\�^�U�T�H�<�<�<�<�<�<�Ϲɹù����������ùϹܹ����������ܹ۹ϼʼ����������������üʼּټݼټ߼ؼؼּʽнνнѽݽ߽������ݽ׽ннннн����������s�Z�N�A�#�,�g����������������������ݿԿѿ̿ȿɿѿݿ������������L�L�Y�]�e�r�y�r�r�p�e�b�Y�L�L�L�L�L�L�L���������w�m�k�z��������������������������	���������	��"�/�;�H�O�T�T�P�H�;�"���������������� ����&�������������������׾ʾ������������׾�����	�������u�l�g�d�o�������ûлܻ���û������A�:�4�-�)�4�4�A�M�M�Z�_�b�Z�N�M�A�A�A�A�a�`�a�f�n�o�zÇÊÇ�|�z�p�n�a�a�a�a�a�a����¸¯«²���������/�<�?�H�@�/�#��
���s�k�g�Z�Z�Z�_�g�s�������������������s�s�����������������'�9�A�5�(����x�s�h�l�s�x�����������ûԻۻջû������x�ּ��������������������������������������������������������������������ֺϺɺƺúɺֺ�������� ������������������������������ ������Z�Y�P�M�F�M�N�Z�f�s���������v�s�f�Z�ZŠŞřŔŗŠŠšŭŮŰŶŹŻŹŭŠŠŠŠ�ܹܹع۹ܹ�����ܹܹܹܹܹܹܹܹܹܹܽ����������������½Ľнѽݽ߽޽ݽнĽ������������������������������������������������z�w�t�r�u�z����������������������������������������������������������������s�^�h�s�w�����������������������������sŠŔŐŎŔŠŭŹ������������������ŹŭŠ�ù����������ù̹ϹӹйϹùùùùùùù�ŹůŭŤŭŹ��������������������ŹŹŹŹ�û��������������л���� �����ܻлþ�����������	����
�	���������������������������������ʼּݼݼּԼͼ˼ʼ�����������!�.�G�S���������������y�`�G�ā�z�t�h�g�d�h�vāčĦĳĳįĦĞĚčĉā�_�X�U�]�_�l�x����������x�l�_�_�_�_�_�_�T�L�G�@�G�Q�T�`�m�v�w�t�m�`�T�T�T�T�T�TE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E���
�������������������
��#�*�0�/�+�#����������������������ĿѿԿݿݿܿԿѿĿ�D�D�D�D�D�EEEE*E;ECEPE[EXEOE6E*EED��\�\�\�R�O�C�A�6�*�%�*�.�6�C�O�\�\�\�\�\�����������������������ʾӾ޾۾׾̾ʾ��� B ; G J ; Q   c f e . ^ * i w L = I S -  " ) N , , � Z Z C � V $ J P !  � I V ] 8 D A 5 8   h s 8 D ] Q @ S G B X j O c I   ( � &   h 0 4    P  X  �  �  �  m  �       �  �  �  �  r  �  �  `  �  7  �  �  �  �  W  f  �  6  �  k  -  {  G  1  �  �  `  5  �  �  �    �  J  X  �  �  �      Y  �  {  �  #  M  �  I  �  �  �    5  \  �  �  ,  �  k  s  �  �  BY  BY  BY  BY  BY  BY  BY  BY  BY  BY  BY  BY  BY  BY  BY  BY  BY  BY  BY  BY  BY  BY  BY  BY  BY  BY  BY  BY  BY  BY  BY  BY  BY  BY  BY  BY  BY  BY  BY  BY  BY  BY  BY  BY  BY  BY  BY  BY  BY  BY  BY  BY  BY  BY  BY  BY  BY  BY  BY  BY  BY  BY  BY  BY  BY  BY  BY  BY  BY  BY  BY  �  �  �  �  �  |  S    �  �  N  5  +  �  �  ]    �  -  o  �  �  �  �  �  �  �  �  �  �  �  �  �  n  Y  ?  "  �  �  4    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  O  O  N  G  @  <  9  5  0  #    �  �  �  �    \  6   �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    |  z  w          �  �  �  �  �  �  �  �  j  ?    �  �  �  O    �  �  �  �  �  �  �  �  �  �  �  �  �  s  Q  0    �  �  �  �  p  _  �  �  �  	  	W  	�  	�  	�  	  	�  	S  �  �  �  �  }  q  S  <  %    �  �  �  �  ~  [  8    �  �  �  u  L  &     �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  s  a  O  =  ,  O  �  �  �  �              �  �  8  �  b  �  �  X  �  �  �  �  �  �  �  �  �  �  ~  s  g  [  P  D  8  ,         �  �  �  �  �  �  �  �  �  �  �  j  G  !  �  �  �  �  F   �  �  �  �  �  �  �  �  �  }  r  f  Z  K  9  '         �   �  S  R  P  P  Q  L  F  :  +      �  �  �  �  l  W  ;  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    s  f  Y  K  >  0  X  c  e  `  V  K  ?  1  !    �  �  �  �  `  3    �  �  �  �  �  �  �  �  �  |  e  M  5    �  �  �  �  o  Q  .     �  F  �  �  �  �  �  �  m  Q  2    �  �  �  k  w  ,  �  j  �  �  �  �  �  |  ^  C    �  �  �  b  0    �  �  �  J  
  �  H  �  �  	7  	b  	�  	�  	�  	�  	n  	N  	'  �  �    g  �  �  �    	  	�  	�  
  
5  
P  
Q  
A  
(  
  	�  	�  	3  �    ^  �  �  T  �  �  �  �  	    %    �  �  �  h  $  �  }     �  0  [  *   �  Z  }  �  �  �  �  �  �  �  �  �  �  t  d  Q  >  *  �  {  �  �  �  �  ~  q  d  R  =  (    �  �  �  �  {  j  Y  F  1    4  �  �  ?  q  �  �  	  	)  	)  	  �  v    �    �  �  _  =  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  h  Q  6    �  K  ;  +      �  �  �  b  E  P  ;    �  �  �  �  h  ?    X  R  @  &    �  �  �  �  t  L  '    �  �  �  �  V     �  >  =  6  *      �  �  �  �  �  n  O  -    �  �  L   �   �  1  (        �  �  �  �  �  s  W  ;    �  �  �  �  p  L  ;  a  x  �  �  �    s  `  A    �  �  �  F    �  P    �  �  =  d  s  s  n  c  X  M  >  )  
  �  �  �  v  $  �  �  /  �  �  k  X  F  3  !      �  �  �  �  d  .  �  �  a     �  Q  E  5  '          #  (      �  �  �  U  !  �  ]   �  P  �  �  �  �  �  �  �  k  I  $  �  �  \  �  �    �  �  �  �  �  �  �  r  d  K  1      �  �  �  `  3    �  �  T   �  W  N  E  <  4  ,  $      �  �  �  |  I    �  �  �    _  �  �  �  �  �  �  2  m  Z    �  ?  �  �  a  �  D  �  �   p  �  �  |  u  l  _  L  8      �  �  �  �  ^  6    �  �  �  �  �  �  �  �  �  �  {  b  @    �  �  c    �  +  �   �   D  Q  P  J  ;  $    �  �  k  *  �  �  L  �  �  h  I  �    (  �  �  �  �  �  �  �  �  g  7  �  �  C  �  O  �    j  �  4  �  �  �  �  �  �  �  �  �  �  �  u  i  ]  P  :      �  �  �  �  �  �  �  �  �  �  h  @    �  �  �  o  @    �  �  �  F  H  E  ;  +    �  �  �  �  �  �  r  T  9  !  �  �  c  
  �  �  �  �  �  �  �  o  ]  I  4    �  �  �  �  j  C     �  �  �  �  �  �  �  �  �  w  j  ^  Q  D  4      �  �  �  �        �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �            �  �  �  �  �  �  o  F    �  �  �  R     �  #            �  �  �  �  �  �  �  �  �  �  }  r  h  ^  �  �  �  �  �  �  {  W  .  
  �  �  �  �  Z  0    �  �  �  y  �  �  �  t  ^  C  !  �  �  �  Z  $  �  �  �  T    �    �  �  �  �  �  �  z  X  1  �  �  �  G    �  �  n    �  b    �  �  �  �  �  �  �  �  �  �  r  \  E  -       �  &  Y  M  y  �  �  �  �  �  �  �  �  r  ;    �  �  M    �  �  d  
  �  �  �  �  �  �  h  N  5    �  �  �  �  �  m  P  3    �  �  �  �  �  �  �  �  �  �  {  s  f  Q  4    �  �  >   �  !          �  �  �  �  �  �  �  �  �  p  T  5    �  �     �  �  �  �  �  �  �  a  4    �  �  _    �  t  �  B  �  �  �  �  �  �          �  �  �  �  s  <  �  �    �  3  k  �  �  �  �  �  �  �  �  �  m    �  Z  �  Z  �  �  �  d  �  �  �  �  �  �  �  �  �  v  [  <    �  �  V  �  �      m  r  v  m  d  X  K  <  ,    
  �  �  �  �  u  K    �  �  �  |  Q  \  �  �    `  ?    �  �  �  �  \  2    �  M  �  �  �  �  �  �  �  �  �  �  �  r  U  /    �  �  9  �  Y   �    �  �  �  t  B    �  �  5  �  �  e    �  K  �  -  P  w  �  �  '  R  p  |  d  6  �  �  x  $  
�  	�  �    !  �      �  �  �  �  w  j  ^  Q  E  7  (    	  �  �  �  �  Y    �  �  �  �  �  |  W  /  	  �  �  �  �  ^  %  �  �  [  �  d  �