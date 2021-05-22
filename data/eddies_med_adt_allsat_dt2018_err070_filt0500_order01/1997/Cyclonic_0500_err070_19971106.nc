CDF       
      obs    ?   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?tz�G�{   max       ?��hr�!      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�O   max       P�^h      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ���   max       <���      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?��R   max       @E�=p��
     	�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��\(�    max       @A��G�|     	�  *x   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @*         max       @O�           �  4P   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�(           �  4�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �J   max       <u      �  5�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��C   max       B,��      �  6�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�y�   max       B,,P      �  7�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >�t�   max       BT�      �  8�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >�K�   max       B�;      �  9�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          r      �  :�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          G      �  ;�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          A      �  <�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�O   max       P�̰      �  =�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��c�	   max       ?ϚkP��|      �  >�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��1   max       <���      �  ?�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?��R   max       @E�=p��
     	�  @�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��\(�    max       @A�33334     	�  Jx   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @*         max       @L@           �  TP   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @��@          �  T�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         DD   max         DD      �  U�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�!-w1��   max       ?ϓݗ�+k     �  V�            q   9   $               O   B      S            "   6   >      (   	   	               !   8      #            !      .      4         
            *                  >      3         1               N��N��*N�U�P�^hOʁcP}N�ݖOG�O:cO��P�̰Pi�_O�xP���N�O84�N�ʏOoS�O;P<iNK-O���Ni)�OE�OQ �O=�O��gO�[O�{MO��sM�OP��O�N��$O�/*P,�OéP
�rN3|�P,UN�o�N���N�N�#�N�R�N�'O�M}O/��O��N/
O�BoO1��P��O�cP�O9 N��O܃O��O�$N�d�N��#O���<���<o;��
;D��;D�����
���
�ě��#�
�#�
�#�
�#�
�T���u�u���
��1��1�ě����ͼ�������������/��/��/��`B��`B��h���������o�+�\)�\)����w�#�
�#�
�,1�,1�H�9�L�ͽT���q���u�u�}�}󶽅������7L��7L��C���C���������������
���T���#/<<==<;/&#��������������������#/0<HMIH=<9/# ���������������}}�)BFJLUXY_ZOB6)
)5:9:MONIB)NNW[]eggg`[QNJHINNNNDOS\goty������th[OCD������������������������������������������#<������{nI<��y������	��������zy�����������������������0Ibx|trbI<#������������������������#/<HU_VUPH</#	
#/00/-#
		����������������|~�������������.HUaz������aULB6.').,/4;><;6/,$#,,,,,,,,z����������zy{zutqrz��������������������stw������������}ytts����������������������
��������
)BW[gpi^a[N5

#/6;:81/#������	 ��������u{��������������wrru�������������������������������������)5BGNPNNB75)#).5BNXZNB;5)%######x{����������������}x$)6[n{���h[OKBB<6$56?BO[bbb\QOB6467865����������������15BEFDB54/1111111111&6[amstsqh[SOB>6*%!&')6=>@<:6*)##''''''
)45:55))




X[hmnhb_[[[SXXXXXXXXMNR[^glt|����tg][NMM����������������������������������������t������������}xnllot��������������$"��������BNW[\df[NKBBBBBBBBBB��������������������
#/342/#
������09;5"�����QT\`amotyzzxpmaXTRPQ�
#/<FB@=4#
����������")*5)#�����>BNUX[[[NB56>>>>>>>>{������������������{fkot����������ytlgef�������������������fgt}�������tgeffffff��������������������anz������������naZ[a�����������������������	������������������ݿֿܿ׿ݿ����������������������������������������������������������˻������!�B�x���Ȼܻ�ܻû��x�l�:�!��!����!�-�:�F�S�_�x�����������z�l�-�!�Z�A�(����������(�5�N�g�t���������Z�������������������ĿɿѿտѿĿ����������#���
�������������
��#�2�=�8�6�2�/�#�x�o�l�_�\�S�E�C�F�S�_�l�x����������~�x�Z�P�M�L�M�U�Z�i�s���������������s�f�Z�����������V�C�*�?�Z�������������������𿫿��y�j�g�z��������ѿݿ�������������������*�6�C�L�O�Y�O�D�6�*����ݽ������������о�(�A�M�L�F� ��ٽ���ݽ:�8�3�:�G�L�S�T�S�G�:�:�:�:�:�:�:�:�:�:���� �#�&�"�)�6�7�B�N�\�U�N�B�@�6�)�ùøìàÓÍÓØÞàìîù��������üùù�����������������������������������Ěčć�|�}ĀĄčĦĳĿ������������ĳĦĚ�f�G�H�T�i�������ʾ׾������׾�����s�fĿĽĿ������������������ĿĿĿĿĿĿĿĿÒÔÜèìù���������������������ìàÒ�ʼɼ��������ɼʼʼּ�ּܼ̼ʼʼʼʼʼ��	�������������	��"�/�3�3�:�/�"���	ìàÖÑËÇÆÇÓàèìïú��������ùì�Z�Z�N�I�G�O�Z�]�g�s�s�������������s�g�Z�;�/�'�'�&�/�2�*�;�H�T�a�k�p�{�s�e�a�T�;�H�C�>�@�H�U�a�n�z�|À�z�t�n�a�U�H�H�H�H�T�O�;�5�.�)�.�;�G�`�m�y���������w�m�`�T��ֻܻǻͻлܻ��"�'�4�?�4�0�&�����Y�T�M�G�M�X�Y�_�f�i�f�^�Y�Y�Y�Y�Y�Y�Y�Y�L�B�@�B�L�m�r�����º˺̺��º������r�Y�LŔŔŇŅŁ�ŀŇŒŔŠŠũŭŭŬũŢŠŔ�h�`�[�W�T�U�[�h�m�q�t�v�w�t�h�h�h�h�h�h�;�/���� �#�/�;�H�T�a�j�q�r�j�a�T�H�;���������������ʾ���>�O�;�.�"�	���ʾ��m�k�`�\�U�U�`�m�y�����������������y�q�m������������ּ�����������ʼ���������������������������������������������v�{�������������������	�����������;�:�3�;�H�T�a�m�u�m�m�a�T�H�;�;�;�;�;�;��
���������)�)�,�)�"������ù����ùϹ۹ܹ���ܹϹùùùùùùùþ���������������	����	���������þ����������������������������������������������$�&�+�-�-�&�$�����q�n�m�p�p�tāčĞĬ������������ĳĚā�q�ݽؽӽݽ�����������������ݽG�F�N�`�������Ľ̽ӽĽ��������y�l�`�S�G�6�+�*�)�*�6�C�F�F�C�6�6�6�6�6�6�6�6�6�6�=�$������'�0�I�b�o�{ǄǊǆ�{�o�b�=���z�s�g�^�s�|�������������������������������f�Y�@�8�6�9�@�M�Y�r����������������	���������������	���"�/�1�4�/�)�"��	�Ŀ��������Ŀѿ����2�G�L�5�"����ݿѿ��z�r�n�d�d�j�n�z�zÇËÓÙàèçàÓÇ�z�n�j�b�n�zÆÇÇÑÒÇ�z�n�n�n�n�n�n�n�n�#�"���������#�<�J�L�W�\�f�h�U�0�(�#²¦¦²¿����������������²�������������������������������������������������������������������������������������������ûлܻ޻ܻлȻû��������������������Ϲ����3�9�9�3����Ϲù� , N P R F X O D D X T 5 W U Q K : H % 8 5 C 0 # ` 4 6  0 ! H W H g T O D l K J E 2 O p S b M 6 e O  X 7 N 3 + . & E o Q q i    �  �  �  i  �  �  �  �  �  N  :  �  y  �    �  �  �  �  0  h  e  �  T  �  �  �  2  h  �  "  n  ^  �  S  j  ;  �  O  �  �  �  K    �  [      �  _  �  �  �  m  �  �  �  �  c    �  �  �<u:�o�o��G��]/�#�
�t���o��/��1��E��������
�Ƨ𼛥�D�����aG����w�� ŽC�����C��\)�Y��ixսe`B�H�9�}󶽰 Žt���o�'8Q�}󶽉7L�@�����8Q콼j�@��ixսT���e`B��hs��+�����\)��E���C����
��1�J���\���
���ٽ��`���罺^5�� Ž�;dBB �5Bb�B}mB��B�2B�zB�B"{B!�kB&�B+||B%�B&s�B+paBM{B��Br�B�$B��A��CB v�B y�B
�.B]�B�iB�B��B�dB��B�B"T3B/B��B��B�B��B,��Bx#By�B�wB`=Bs�B	g{Bs�B�IB
�<BM�Bv�Bp�Bm5Bw�B��A���BgB�HB�B2B
i�B�fB
	�B*8TBסBB>�B>RB�B��BҸB��B=�B"�B">�B'GgB+�	B<B&��B+E�B<B�>B@	B��B�A�y�B � B PB
�TBB��B�(B��B��B��B�B"~�B?BAB=�B��B��B,,PB@�BH�B�-Bx�BE0B	�oB��B:�B
��Bk�B@zBwB@�BV�B��A�vCB�Bf�B>�B=�B
A8B>B
=�B*O�B=iA��|A/-A���@��X@��A���AwU_A��@�XZACS�A��Av��A���A,i�A��Aג�A��A�k A�AJ]�A� �Aί�@�zSA�~�A��~A��uA��BA�"�Ai%I@�L�@��@�\A�A�W7A���AS�ZAm�@��A�e�A� A�h�A��2>�t�AY�uA�<�B	SAA�X�A/K�A�ZB \�BT�A��@䱯A���A��A�n@A�ǚA��A���A�ɮA���@� ?d8A���A~�hA���@�%�@���A��sAvU+A��4@�?AD�[A�_�Avx�A�x�A-��Al^Aׁ�A�_A�~yA�eAJ�A�oA�Qd@��A���A�v�A��kA���AŜ8Ah��@��@�D?�B�A�fAۀA��AN��Ak'�@�3�A�~�A���A�rA���>�uHAZ��A�x`B	@MAފA1�6A�B =B�;A��@��ZA��A×A�2�AȁVA�k�A�heA�A�A���@��.>�K�            r   :   $               P   B      S            #   7   >   	   (   	   	               "   9      #            !      .      5                     +                  ?      3         1      	                     ?   !   +               A   5      G               !   -      !               !         !      /         #   3      1      '                     #      #      #      )      '         %               '            9                     A   1      -                  !                                             #   -      1      #                                       %      !                        'N��N��*N�U�P�,OY��O���N�ݖOG�O:cO��P�̰PK�mO�xPt�N�O�
N�-�N��JO_O�NK-O
�Ni)�OE�OQ �O5<�O�iN��OR"�OJ}M�OO�y�O�N��$O�/*P��OéP
�rN��O�yAN�o�N�)N�N�#�N�R�N�,�Oy�vO/��Oo|�N/
N�@�OQP�"O�cO�XjOsZN��O��lO��O�$N�d�N��#O���  �    _  	�  V  s  i  p  �  ]    B  �  �  �  �    )  �  \  ]  |    ?  �  �  �  �  l  w  %  �    J  �  T  n  G  |  �  �  �  #    �  P  �  �  �  5  [  �    �  �     �  �  5  o  �  M  �<���<o;��
��o�u�T�����
�ě��#�
�#�
�#�
��o�T���t��u�ě���9X�\)�'<j�����'�����/��/��`B�+�o�t��D�����'��o�+��P�\)���#�
�T���#�
�49X�,1�H�9�L�ͽY���C��u��o�}󶽍O߽�O߽�C���7L��t���hs��C���1�����������
���T���#/<<==<;/&#��������������������#/0<HMIH=<9/# ��������������������&)6:BKPSTROKB6)! &(5BGJJIDB5)NNW[]eggg`[QNJHINNNNDOS\goty������th[OCD������������������������������������������#<������{nI<����������������}}���������������������0<ISbjpsmbUI<1#��������������������#/<GHQMHD</)##..,#��������������������������������8HUgnxz���zthaUH9648,/4;><;6/,$#,,,,,,,,yz�������������zzyxy��������������������stw������������}ytts�����������������������	�������)5BKNXPQNB5)#/2773/%#"��������� ���������}���������������|||}��������������������������������������)5BGNPNNB75)#).5BNXZNB;5)%######x{����������������}x)6Yly����d[OFE>6& )56?BO[bbb\QOB6467865����������������25BDDBB5502222222222)06O[ckmlhhd[OB6/-()')6=>@<:6*)##'''''').53)%X[hmnhb_[[[SXXXXXXXXMNR[^glt|����tg][NMM����������������������������������������t{������������tqpppt��������������� ������BNW[\df[NKBBBBBBBBBB��������������������	
"#',/01/+#
	����.7983 �������QT\`amotyzzxpmaXTRPQ��
#/:<931#
���������')) �����>BNUX[[[NB56>>>>>>>>��������������������fkot����������ytlgef�������������������fgt}�������tgeffffff��������������������anz������������naZ[a�����������������������	������������������ݿֿܿ׿ݿ����������������������������������������������������������˺�������!�F�x�������ûѻ׻ػл��l����-�*�*�-�9�F�S�_�l�x���������w�l�_�F�:�-�A�5�(������(�5�A�N�g�j�t�w�z�s�Z�A�������������������ĿɿѿտѿĿ����������#���
�������������
��#�2�=�8�6�2�/�#�x�o�l�_�\�S�E�C�F�S�_�l�x����������~�x�Z�P�M�L�M�U�Z�i�s���������������s�f�Z�����������V�C�*�?�Z�������������������𿒿y�n�j�k�z���������ѿ������鿸������������*�6�C�L�O�Y�O�D�6�*������������������Ľݾ��4�;�;�1��ݽĽ����:�8�3�:�G�L�S�T�S�G�:�:�:�:�:�:�:�:�:�:�)�"��#�&�)�*�6�B�J�O�X�R�O�K�B�;�6�)�)àÚÙàìù��������ùìàààààààà������������������������������������ĚĘČĉċčĔĚĦĳ������������ĿĳĦĚ��g�d�s�w����������¾ʾ׾۾پо������ĿĽĿ������������������ĿĿĿĿĿĿĿĿìèåêìùÿ����������������������ùì�ʼɼ��������ɼʼʼּ�ּܼ̼ʼʼʼʼʼ��	�������������	��"�/�3�3�:�/�"���	ìàÖÑËÇÆÇÓàèìïú��������ùì�g�Z�N�K�H�N�P�Z�_�g�s�~�������������s�g�;�/�+�*�*�*�/�2�;�T�a�a�m�v�s�n�a�T�H�;�H�F�B�E�H�U�a�n�u�z�{�z�n�n�a�U�H�H�H�H�`�T�K�G�A�<�>�G�T�`�m�y�{�������y�m�k�`���ܻ׻ѻֻܻ����'�'�%��������Y�T�M�G�M�X�Y�_�f�i�f�^�Y�Y�Y�Y�Y�Y�Y�Y�L�G�I�L�Y�e�r�~�����������������r�e�Y�LŔŔŇŅŁ�ŀŇŒŔŠŠũŭŭŬũŢŠŔ�h�`�[�W�T�U�[�h�m�q�t�v�w�t�h�h�h�h�h�h�;�/���� �#�/�;�H�T�a�j�q�r�j�a�T�H�;�������������ʾ���5�9�.�"�	���ʾ������m�k�`�\�U�U�`�m�y�����������������y�q�m������������ּ�����������ʼ������������������������������������������������|�����������������������������������;�:�3�;�H�T�a�m�u�m�m�a�T�H�;�;�;�;�;�;���������'�)�+�)�!��������ù����ùϹ۹ܹ���ܹϹùùùùùùùþ���������������	����	���������þ����������������������������������������������$�&�*�-�,�$�$�����|�t�r�q�u�u�|āĊĚĦĳĿ��ĿĳĦĚč�|�ݽؽӽݽ�����������������ݽS�K�I�Q�`�l�y�������������������y�l�`�S�6�+�*�)�*�6�C�F�F�C�6�6�6�6�6�6�6�6�6�6�b�\�V�I�C�C�I�V�b�o�{�~Ǆ�~�{�o�b�b�b�b�������t���������������������������������f�Y�B�9�7�@�M�Y�f�r�����������������f�	���������������	���"�/�1�4�/�)�"��	�ȿ������ſѿ�����,�?�@�5�(�����ݿ��z�v�n�f�f�l�n�zÆÇÓ×ààçäàÓÇ�z�n�j�b�n�zÆÇÇÑÒÇ�z�n�n�n�n�n�n�n�n�����
���#�0�<�I�L�U�Z�X�I�<�0�#�²¦¦²¿����������������²�������������������������������������������������������������������������������������������ûлܻ޻ܻлȻû��������������������Ϲ����3�9�9�3����Ϲù� , N P Z = A O D D X T 4 W K Q D > K  " 5 ; 0 # ` /  "  $ H N H g T I D l < > E , O p S [ : 6 T O D D 3 N / ( .  E o Q q i    �  �  �    �  t  �  �  �  N  :  r  y  �    K  �  �  �  �  h  7  �  T  �  �  	  �  �  �  "  J  ^  �  S    ;  �    �  �  �  K    �  2  	      _  �  *  r  m    U  �  #  c    �  �  �  DD  DD  DD  DD  DD  DD  DD  DD  DD  DD  DD  DD  DD  DD  DD  DD  DD  DD  DD  DD  DD  DD  DD  DD  DD  DD  DD  DD  DD  DD  DD  DD  DD  DD  DD  DD  DD  DD  DD  DD  DD  DD  DD  DD  DD  DD  DD  DD  DD  DD  DD  DD  DD  DD  DD  DD  DD  DD  DD  DD  DD  DD  DD  �  �  �  u  b  K  ,    �  �  �    X  1  
  �  �  �  r  Z              �  �  �  �  �  �  �  �  �  }  c  G  *    _  R  E  7  (    
  �  �  �  �  �  �  �  �  �  �  �  u  e  �  	C  	�  	�  	�  	�  	�  	v  	8  �  G  �  �  C  �  �  2  E  �  X  U  �  �  �  )  H  V  Q  A  "  �  �  ~  7  �  i  �  �  
  �  	  +  @  Y  q  q  m  n  l  j  m  h  I    �  �  !  �  i  o  i  _  U  K  A  7  .  $        �  �  �  �  �  �  �  �  �  p  n  l  i  d  [  Q  H  >  3  %        �  �  �  �  �  `  �  �  �  x  g  U  B  .    
  �  �  �  �  c  >    �  �  [  ]  N  =  (    �  �  �  �  w  [  F  6    �  �  �  p  F      �  �  �  g  I  )    �  �  �  X    �  w    �  8  ^   �    A  B  A  ?  ;  .    �  �  �  H    �  L  �  v    l  �  �  �  �  �  �  �  �    {  v  p  j  b  Y  Q  K  D  4    
  �  �  %  v  �  �  �  }  e  P  >  D  j  l  R  �  u  �  D  r  �  �  �  �  �  �  �  �  �  �  �  �    Q  �  d    �  �  5  (  P  �  �  �  �  {  c  B    �  �  m  "  �  d  �  �    �  �  �          	    �  �  �  �  �  �  �  �  �  �    y    R  �  �  �  �      )  '    �  �  �  G    �  �    �  <  j  �  �  �  �  �  �  �  �  �  \    �  }  �  Y  �  &  ]  �  �    5  I  R  Y  [  L  1    �  �  �  .  �  S  �  �  F  ]  F  .    �  �  �  �  �  `  ;    �  �  �  j  @    �  �  u  �  �  �  %  G  f  x  y  k  G    �  �  3  �  �  ?  �  S        �  �  �  �  �  �  �  �  �  q  P  /    �  �  �  �  ?  9  4  +  !              �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  e  A    �  {  /  �  �  j  �  V  �  �  �  z  d  I  '    �  �  |  A     �  c  	  �  *  �  Z  m  r  x  �  �  �  �  �  x  V  +  �  �  �  s  O  &  �  i  �  �  �  �  �  �  �  �  �  �  n  S  >  )    �  �  <  �  �  ]  :  G  S  d  l  j  c  T  @  &    �  �  �  [    �      �  �    0  D  V  m  w  r  e  O  '  �  �  L  �  �      n  ^  %  !      	  �  �  �  �  �  �  y  V  0    �  �    R  %  C  O  p  z  �  �  �  �  �  �  o  M    �  �  ^  >  �  �  #    �  �  �  �  �  �  z  f  R  >  (    �  �  �  �  b  @    J  *  
  �  �  �  {  P  "  �  �  �  S    �  V  �  �  =  �  �  �  �  �  t  X  :    �  �  �  \  '  �  �  �  ~    �     C  Q  >        �  �  �  �  �  �  x  a  G  !  �  m  �  e  n  g  ]  O  >  *    �  �  �  �  �  {  O    �  �  }  K    G  5       �  �  �  �  �  G  �    �  �  ]  �  �    �  1  f  m  t  {  u  l  d  ^  Z  V  X  `  h  o  r  u  w  v  u  t    V  s  {  �  �  x  l  X  A     �  �  }    �    �  �  �  �  �  �  }  u  l  a  S  E  1      �  �  �  �  g  G  (  
  �  �  �  �  �  �  q  R  0  	  �  �  m  6  �  �  �  _  /    #  b  �  �  �  �  �  �  �  �  u  \  C  )    �  �  �  �  �      
    �  �  �  �  �  �  �  �  n  X  E  3  !  
  �  �  �  �  �  �  �  �  �  s  >    �  �  �  T    �  w    �    F  M  O  J  ;  )    �  �  �  �  ]  2     �  |    �  =  �  O  \  �  �  �  �  �  �  �  x  S  (    �  v  �  k  �  �  5  �  �  }  l  Z  D  -    �  �  �  �  �  �  t  U  3    �  �  �  �  �  �  �  �  �  �  �  �  �  y  F  �  �  �  j    �  =  5  &      �  �  �  �  �  �  c  5    �  �  m  8    �  �  T  .    �    F  R  W  R  5    �  �  i    �  P  �  A   �  �  l  Y  j  �  s  [  >    �  �  �  d  /  �  �  �  P             �  �  �  n  ;    �  �  e  8  �  �  '  �     �  j  �  �  �  �  �  �  {  X  .  �  �  �  _  *  �  �  z  9  �  �  h  �  �  �  �  �  �  w  ]  <    �  �  a    �  :  �  �    �  �  �  �  �  �  �  �  c  &  �  �  >  �  X  �  Y  �  h  �  �  �  �  �  �  �  n  W  >  %  
  �  �  �  �  [    �  �  3  Y  �  �  �  �  �  �  �  �  �  �  �  n  %  �  /  �  �  �    5  (    �  �    D  
  �  �  r  @  8    �  �  C  �  S  �  o  [  H  1       �  �  �  �  z  ^  B  .        �  �  x  �  �  �  �  �  �  ^  :    �  �  �  v  I    �  �  �  p  ;  M  <  *      �  �  �  �  �  �  �  j  T  =  &     �   �   �  �  �  k  6    �  �  B    �  �  �  �  I    �  �  7  �  �