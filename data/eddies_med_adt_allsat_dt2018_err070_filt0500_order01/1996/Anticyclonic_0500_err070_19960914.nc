CDF       
      obs    ?   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?Ǯz�G�      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M���   max       P���      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �t�   max       =\      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?��G�{   max       @F��R     	�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��\)    max       @v������     	�  *x   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @0         max       @Q�           �  4P   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�x`          �  4�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ���
   max       >]/      �  5�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       B4�   max       B/      �  6�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       B �!   max       B/��      �  7�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?�   max       C���      �  8�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?��7   max       C���      �  9�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  :�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          3      �  ;�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          '      �  <�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M���   max       P0      �  =�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�����+   max       ?��&��IR      �  >�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �C�   max       >
=q      �  ?�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @@         max       @F��R     	�  @�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��\)    max       @v���
=p     	�  Jx   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @(         max       @Q�           �  TP   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�X�          �  T�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         B�   max         B�      �  U�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��S&�   max       ?޽<64     �  V�                  1            	   �            '                        !   =            �                  &            0      -      0                                       
   !      	   J         g   O.G9N�FqO^lN/[�NO��O���O4�NFl�O�|�N^|kP���N�u�O��zN�BOK�NO�ŉN6k�N�!�NyK�O���O�O��O�S�P�NtK N�9`N���Px}�Ns��NB�N���O3j�O$`�P$��O0�0O &�Ob1ZO$��N�O��N���P��N���O'>�O��NM �N�;Om�Oc�XN�zO:�Nb�N'��N�mgN���O''�M���NR'�O���N̭O��O�P�O׽t��e`B��o��o:�o;ě�;ě�<o<#�
<e`B<�o<�C�<�t�<�t�<���<���<��
<�1<ě�<ě�<���<���<�`B<�`B<��=o=o=C�=\)=�P=��=��=�w='�=8Q�=8Q�=@�=D��=D��=D��=H�9=H�9=T��=Y�=Y�=Y�=aG�=aG�=e`B=e`B=e`B=ix�=m�h=m�h=q��=u=u=�+=��P=���=�{=�^5=\VWRY[agot�����xtg`[V1-/46BKOQUVWROHB;611MNPPSV[gmtwzzwtjg[NM��������������������')5BLNSYNB<51)''''''���������
��������������������������������������������1747B[g|������t[NB51�����������������������)5BNgtqg[N5������������������������������+!�������RXY[hqt�������zth[RR#/<GHQUYYUSH/(!������0223540#
�������������������������������������������NN[cgtvtsg][XRTNNNNN��������������������ww����������������~w" #(0<IUbfbUPI<0#"�� )6ADC6NGGJHOh���������th[N#000<GA<0#�����������������������

���������������''�����������������������	%)4)�{negnr{���������������������������������~~���������������������)4:>BA;0)���������������������� %)25BNPYXWNB51)    ��������������������SQPUanrz{��}znea]XS)6>BBBBBEB=6)#42126ABOXhuvvwt[OB94������� 	����������������6?C<)�������������������������
##'*-+&#
��
#$)**&#
	)+2)#)55:5/)A?BEN[fgopngg[NIFCBA����������������������������������������##//<HOU[^^aUH<;/*#jccnz��znjjjjjjjjjj=8<@BHOWONCB========!#07;;40/#txz�����������ztttt��������������������

"��������������������������� 
 
��������������#/38;8/#

###��������

��������*6<>=6*������zÇÓàèìîàÙ��z�n�a�^�U�Q�V�a�q�z���Ľнݽ߽�ݽٽнĽ�����������������������������������������������������������������)�2�0�)�(�����������T�X�[�[�`�b�`�T�J�J�G�E�G�P�T�T�T�T�T�T�6�B�O�V�]�V�K�N�F�6�)� �����"�)�3�6���������ûȻȻû��������������{�~�������T�]�V�T�N�G�;�9�6�;�G�P�T�T�T�T�T�T�T�T��������ʾپܾ׾ʾ���������w�{�y�t�y��������¼ȼ���������������������������������)�H�k�yāąĄ�|�h�)����������������'�3�<�@�L�R�L�G�E�@�3�0�'�#��$�'�'�'�'������3�E�B�4�*������ֿѿ˿ƿ˿̿�@�L�Y�Y�_�e�g�f�e�\�Y�V�L�D�@�=�;�=�@�@��������������������������������(�4�A�M�R�W�Q�M�A�(������������Ľнݽ��������ݽнĽ��ĽĽĽĽĽĽĽ�����������������������������þ�����������[�b�`�[�T�P�N�E�B�>�5�4�5�B�N�[�[�[�[�[�	��"�.�;�G�K�I�;�0�"�	���ؾ̾Ҿ׾�	�n�zÊÍÇ�z�s�zÇËÇ�z�n�a�]�^�V�[�a�n�ּ���������� ����������ּռ־��ʾ׾��	��"�%��
���׾ʾ������������f��������������¾������s�f�\�T�M�L�Q�f�������������������x�s�{�}������������	�����	�����������������������5�6�@�<�5�2�)� ���������)�4�5�5�����лٻ�������л������}�x�v�t�}�����������������������������������������ѻ�������������������������������!�!�!������������(�5�A�N�Z�_�b�d�f�Z�N�A�5�(������(Ƴ����������������������ƺƳƲƭƨƧƮƳ��������5�<�3��������ƳƚƎƁƇƖƬƸ��������������ùìàÓÏÐÓÜàù������������� �%����������������������������������������������������������FJFVFcFnFnFhFcFWFVFJF=F1F$F FF$F(F1F=FJ�������������������v�r�f�[�f�j�m�r�w������'�6�M�W�U�M�D�4�'��������������������������������������������"�/�5�8�3�"������������������Z�g�q�g�_�^�Z�P�N�A�5�(�&�(�5�A�F�N�X�Z����	��"�-�.�5�6�.�"��	���������`�m�y���������{�y�m�`�T�G�<�C�G�T�U�`�`��'�,�3�:�@�3�'�����������������������������������������������������������������������s�g�Z�Z�O�T�Z�g�s����ù����������������ùàÓÇ�z�o�rÃÓàù�U�a�l�j�a�_�U�H�<�/�*�/�7�<�H�N�U�U�U�U��"�+�/�/�,�'�"���	������������� �	���(�4�=�=�4�(��������������������������������������������������������������ɺϺҺɺ������������������������
���#�#�#����
���������� �
�
�
�
ĚĦĳĺĿ������ĿĳĦĚċā�ĀāčĖĚ²¿������¿¿²¬±²²²²²²²²²²�/�<�H�R�H�F�<�/�#�"�#�/�/�/�/�/�/�/�/�/EuE�E�E�E�E�E�E�E�E�E�E�E�E�E�EsEkEiElEu�
���#�(�#��
�	� �
�
�
�
�
�
�
�
�
�
ǭǥǡǔǈ�{�r�p�u�{ǄǈǔǡǬǭǰǭǭǭD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D��y��z�|�}�z�z�r�l�`�S�M�K�O�V�`�g�l�t�y V A 1 � � W   f ' \ 9 = < R < 6 u ; V E r b ? % ~ 3 9 C r l C > [ S e 4 L K D ( . v { 4 H W � = l _ L   F C = > k j d _ + )     �    I  �  �  �  y  �  )  �  ;  �      �  y  �  �  �  �  �  �  w  �  �  �  �    �  e  �  �  �  !  �  0  �  n    �  �  L  7  n  3  x  <  h  5  �  �  g  S  �  �  ~    W  �  ;  J    弣�
��o<t�<���;�o=aG�<�j<49X=C�<�j>L��<�`B=8Q�=o=ix�=8Q�<�/=�w<�h=T��=49X<��=u=�E�=\)=�w=t�>]/=�P='�=]/=m�h=q��=��w=�%=T��=���=\=�O�=�v�=ix�=ě�=e`B=�t�=��=q��=ix�=�O�=�{=�o=��-=�%=�o=���=�C�=�v�=�o=���>�+=�1=�"�>F��=�l�B	��BS�B	�B��B��B/B#
iBF�B	1MB k8Bx�B!\ BTbBB��B$��B�0B�[B�B �B�zB&YB��Bk�B%�GBa]B�PB��BbCB�`B(�B3hB4�B!B!�B��B��B�OBcB�B�xBX�B�B>tB=*B��B�lB��B�B^/B�B�{BX9B%v_B�B�
B�B�B�cB�1B��B�B/B	��B;�B	>�BA�B�lBD�B#%4BB�B	AlB �	B�_B!K�B@�BR�B��B$�1B�`B(�BŎB BD�B%U�B¢BCB%�gB:�B�jB�#B�`BA+B(�PB@B �!B��B!CkB�B��B�,B@�B�,B�1B��B �BI�BAB�B��B��B��B@B7B<kBQ�B%��B�qB�B7dB�B��B�B_�BB/��AȒ]A&2CA� Aե�Ag�hA�k@���Ae��AL+@��YAךZ?�J�A�½?��A҂�A6�A*�AϨ�A�*A\{PA��A�AS�)AFC@� A�uDA���@���A�O@^�U@[��A�dtB�aB�cAͪA��A���C���@�)@ű�A���A��^A��A[½AjW?�A��TA�w�A̝�A��[A���A7�f@��@$V�A���A�8�A�r�A�kC�GA�H�BgC���AtfAǇ{A&��A�rpA�x4Ag!�AׁO@��Ad�!AK�2@�YA�i�?�?�A���?Ԩ�A�{�A7��A*�TA�z�A���A\�jA��A`AS �AF�n@�.A���A���@��A�h�@c�<@\�A���B�MB7�A̡�A�лA�FC���@��@��A�~�A�\�A��A["�Aj�?��7A��A�`AA�~�A��A���A8��@!�@%YA�~A�~�A��uAª�C��A�tB(�C��A�                  2            
   �            '                        !   >            �                  &            0      .   	   0                        	               
   "      
   J         h                              #      1      %                              !   '            1                  /                        3                                                                                                '                                                                     !                                                                                       O5rN�PN�e�N/[�NO��NH N�/ENFl�Oc'N^|kP0NIXxO���N�-�O��O���N6k�N���N��O�=&N%P�O��O0�
O��yN9�^N�9`N2��O�D�Ns��NB�N���O4Ob�O���O0�0N�OS�O/&N�G�OE�N���O���N���O'>�N�f�NM �N�;O�'O/\*N�zO:�Nb�N'��N�mgN���OL.M���NR'�O	��N̭O��N��Oz�  &  y  [  `  �  �  }  �  �  �  �  f      �  �  �  �  7  H  �  	  �  O  )  �  �  �  �  T  �  L    :  ,  {  �  D  �  G    c  q  �  �  �  �  �  O  �  >  S  �     �  �  �  �    �      Y�C��T���o��o:�o=�P<#�
<o<��
<e`B=��T<��
<�j<��
<���<�1<��
<�9X<���<�/=+<���=�P=@�=o=o=C�=�=\)=�P=��=#�
='�=P�`=8Q�=<j=D��=P�`=L��=}�=H�9=}�=T��=Y�=]/=Y�=aG�=e`B=u=e`B=e`B=ix�=m�h=m�h=q��=}�=u=�+=��=���=�{>
=q=ě�Y[[_gkty����vtogb][Y3.056BGOPSUUPOKB>633QRSVZ[ghtuxwutg[QQQQ��������������������')5BLNSYNB<51)''''''������������������������������� ���������������������������MHFGN[gt|����ztg[VNM��������������������
5BN[gmnl_WNB5����������������������������
�������]]aht����vth]]]]]]]]#/<@HMUUTLH</+%##������#-012430#
������������������������������������������ZTVS[`gjng[[ZZZZZZZZ����������������������������������������" #(0<IUbfbUPI<0#"
)156986.)TPQSX[h����������h[T #0<D?<0#�����������������������

�������������������������������������������	%)4)�{negnr{�������������������������������������������������������),357783)����������������������())45BMNWWTNB65)((((��������������������TRRUanqzz~��{znga^YT)6<A@=6)%=<;::>BO[hlpnhd[WOB=������� 	�����������-7<<95)��������������������������
##'*-+&#
��
#$())%#
	)+2)#)55:5/)JFDCDN[bgnomgf[NJJJJ����������������������������������������##//<HOU[^^aUH<;/*#jccnz��znjjjjjjjjjj=8<@BHOWONCB========!#07;;40/#txz�����������ztttt��������������������

"��������������������������

���������������#/38;8/#

###�������

����������*6;><6*����zÎÓàçàÔÇ�z�n�a�`�U�S�U�X�a�n�w�z���Ľнݽ޽߽ݽԽнĽ�����������������������������������������������������������������)�2�0�)�(�����������T�X�[�[�`�b�`�T�J�J�G�E�G�P�T�T�T�T�T�T�6�B�E�C�B�@�6�)�'�'�)�-�6�6�6�6�6�6�6�6�����������ûĻĻû����������������������T�]�V�T�N�G�;�9�6�;�G�P�T�T�T�T�T�T�T�T���������ž;ӾѾʾ����������������������������¼ȼ���������������������������������)�6�j�s�v�u�j�[�O�6������������'�3�5�@�E�B�@�3�'�&� �'�'�'�'�'�'�'�'�'�����*�@�8�)�!�������߿ֿҿ׿ݿ���@�L�Y�\�e�d�Y�L�F�@�?�=�@�@�@�@�@�@�@�@�������
��������������������������(�4�A�M�P�U�Q�M�?�4�(�����������Ľнݽ��������ݽнĽ��ĽĽĽĽĽĽĽ�����������������������������ÿ�����������5�B�N�[�^�[�O�N�B�@�5�5�5�5�5�5�5�5�5�5�	��"�.�;�B�H�E�;�.�(�"��	��߾׾��	�n�n�z�t�n�o�n�m�a�a�a�g�a�`�a�n�n�n�n�n�ּ���������� ����������ּռ־��ʾ׾�����������׾ʾƾ����������s������������������������s�p�f�^�[�e�s�������������z�v�}����������������	�����	�������������������������#�)�5�7�5�-�)�%���������������ûлܻ���ܻлû��������������������������������������������������������ѻ�������������������������������!�!�!������������(�5�;�A�N�Z�]�`�a�Z�X�N�A�5�+�(���"�(Ƴ����������������������ƶƳƯƪƩưƳƳ��������������������ƳƧƛƚƦƹ����������������ùìàÓÏÐÓÜàù��������������#���������������������������������������������������������FJFVFcFiFmFgFcFVFUFJF=F1F$F"FF$F*F1F=FJ�����������������r�n�o�r�x������������'�4�;�D�B�:�4�'��	����������������������������������	�"�/�1�1�,�"��	���������������������	�Z�g�q�g�_�^�Z�P�N�A�5�(�&�(�5�A�F�N�X�Z����	��"�-�.�5�6�.�"��	���������`�m�y���������{�y�m�`�T�G�?�G�G�T�V�`�`��'�,�3�:�@�3�'�����������������������������������������������������Z�g�s���������������s�g�\�Z�P�U�Z�Z�Z�Zìù��������������ùàÓÇ�z�zÇÉÓàì�U�a�l�j�a�_�U�H�<�/�*�/�7�<�H�N�U�U�U�U��"�+�/�/�,�'�"���	������������� �	���(�4�=�=�4�(��������������������������������������������������������������ɺϺҺɺ������������������������
���#�#�#����
���������� �
�
�
�
ĦĳĸĿ����ĿľĳĦĚčČāĀāāčĚĦ²¿������¿¿²¬±²²²²²²²²²²�/�<�H�R�H�F�<�/�#�"�#�/�/�/�/�/�/�/�/�/E�E�E�E�E�E�E�E�E�E�E�E�E�E}EuEsEsEuE�E��
���#�(�#��
�	� �
�
�
�
�
�
�
�
�
�
ǭǥǡǔǈ�{�r�p�u�{ǄǈǔǡǬǭǰǭǭǭD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D��l�x�y�{�|�y�y�q�l�`�S�N�L�O�S�W�]�`�i�l T E 1 � � J * f  \ 5 C - S 6 4 u 6 T : � b   $ b 3 Z + r l C ; K = e + H H ! / . + { 4 F W � ' b _ L   F C = ; k j = _ +  n    X  �    �  �  m    �  �  �  �  m  H  �  K  D  �  �  S    �  �  r  T  �  �  l  �  �  e  �  X  L  �  �  �  �  U  �  �  �  |  7  n     x  <  #  �  �  �  g  S  �  �  U    W  /  ;  J    z  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�    !  %  "      �  �  �  �  �  �  �  �  �  �  n  K    �  m  s  y  y  x  q  k  f  b  d  ]  M  <  *    �  �  �  �  _  F  R  X  Z  S  E  4      �  �          '  1  F  y  �  `  	  	�  
A  
�  
  r  �  @  �  ,  �  	  f  �  �  �    B  g  �  �  �  �  �  �  �  �  �  �  }  l  [  J  :  *    	   �   �  �  �  �  �    K  g  �  #  L  g  ~  �  p  2  �  /  �    q  W  e  o  v  {  |  u  d  N  6  %        	  �  �  �  �  H  �  �  �  �  �  �  �  �  �  �    q  c  N  +     �   �   �   {  @  W  l  �  �  �  �  �  �  �  �  �  ~  N    �  p  �  _   �  �  �  �  �  �  �  �  �  �  |  e  L  2    �  �  �  �  k  :  �    �  �  %  �  �  �  �  [  �  d  �  �  �  f  �  	v  9  �  "  2  B  P  [  d  e  c  S  A  .      �  �  �  �  a  3    �  �  �  �     �  �  �  �  �  �  �  f  H  $  �  �  O  �  Q  �          	     �  �  �  �  �  �  �  �  ~  o  V  5  
  �  �  �  �  �  �  z  p  ^  L  >  /    �  �  '  �  $  �  �  �  �  �  �  z  g  M  -    �  �  z  E  
  �  �  X    �  j  �  �  �  �  �  �  �  �  �  o  ]  K  7  #    �  �  �  {  K  �  �  �  �  �    m  U  7    �  �  �  �  h  ?    �  �  {  6  6  7  7  5  *         �  �  �  �  �  k  N  2     �   �  #  8  G  D  ;  .        �  �  �  �  �  �  K    �  q  �    �  �  �  +  e  �  �  �  "  �  e    �  �  a  5    �  �  	    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  c  �  �  �  �  �  �  �  �  �  �  {  T  '  �  �  @  �  �   �  ;  �  �    0  J  O  L  C  9  (  �  �  �  I  �  u  �  <    #  %  &  '  (  )  &  $  !            �  �  �  �  �  �  �  �  �  �  p  ^  I  2       �  �  �  �  r  S  5  5  B  O  v  u  t  s  r  r  v  {    �  �  �  �  �  �  �  �  {  t  l  B  �  Y  �  �  �  >  s  �  �  c    �  �  )     
�  	  J   �  �  �  �  �  }  w  q  k  e  _  W  N  E  <  2  )           T  F  9  +      �  �  �  �  �  {  Q  '   �   �   �   �   {   [  �  �  �  x  ^  D  *    �  �  �  �    ?  �  N  �  D  �  I  9  H  K  @  3  $    �  �  �  �  ]  $  �  �  b    �  0  �  �  �  �  �  �  �  �  f  <    �  �  u  7  �  �  a    �  �  �  �    !  3  :  0    �  �  �  T    �  �  d    �    n  ,  $  �  �  �  T  $  �  �    2  �  �  O     �  �  �  U  �  y  z  z  x  o  f  \  R  G  9  +      �  �  �  �  �  k  E  �  �  �  �  �  �  �  n  K  !  �  �  �  [  %  �  �  "  �  8  =  D  C  :  *    
�  
�  
�  
@  	�  	p  �  m  �  O  �  �  �  �  4  R  �  �  �  {  b  F  *    �  �  �  M  	  �  w  3  �  �  �  �    *  <  E  G  B  1    �    �  �  �  @  �  T  �  }        �  �  �  �  �  �  �  �  �  �  e  E    �  �  |  A  E  +     4  Q  H  L  !  �  �  �  L     �  b    �  5  �  �  q  m  i  e  b  Z  J  9  (      �  �  �  �  �  �  �  �  t  �  �  �  �  �  �  u  d  S  =  !    �  �  �  U    �  2  �  �  �  �  �  �  �  v  [  >    �  �  �  N      �  �  �  b  �  �  y  g  ^  V  O  H  A  ;  ;  C  J  E  &    �  �  �  W  �  ~  t  j  _  U  K  A  7  -  $        	    �  �  �  �  �  �  �  �  �  �  n  S  5    �  �  �  t  6  �  �  -  �  t  .  <  C  N  J  F  7    �  �  z  >  �  �  Z  �  6  N  U  $  �  y  p  k  i  f  a  \  T  K  >  -    �  �  n  �  h     �  >  &    �  �  �  �  �  �  �  �  �  Y  -    �  �  w  �  �  S  B  1       �  �  �  �  �  �  a  9  *  L  m  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  o  _  N  <  )    �  �  �     �  �  �  �  �  �  �  o  U  :       �  �    \  {  �    �  �  �  �  �  �  �  �  �  o  P  0    �  �  �  X     �  �  �  �  �  w  S  '  �  �  �  T    �  l    ~  �  +  x  �  *  �  �  �  �  �  �    k  V  A  @  R  e  x  �  �  r  a  Q  @  �  �  �  �  �  �  �  z  d  M  5      �  �  �  �  a    �  �  V  U  �  �  �      �  �  F  �  L  
�  	�  �  �  �  <  a  �  �  s  [  C  )    �  �  �  �  �  �  �  �  �  h  M  2      �  �  �  Z  .    �  �  l  3  �  �  W    �  W  �  �  �  �  �     g  �  �  �         �  �  3  �  �  �  �  �  �  �  �  U  F  &    �  �  �  m  B    �  �  �  M  !    �  �  