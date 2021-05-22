CDF       
      obs    L   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�G�z�H     0  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�i6   max       P�R:     0  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��E�   max       =�w     0      effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?0��
=q   max       @FaG�z�     �  !<   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��(�\    max       @v|(�\     �  -   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @,         max       @P            �  8�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @͗        max       @��@         0  9�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �"��   max       <���     0  :�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�ۥ   max       B0�{     0  ;�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A���   max       B07     0  =$   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >���   max       C��M     0  >T   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >��w   max       C��     0  ?�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �     0  @�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          9     0  A�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          5     0  C   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�i6   max       Pm�@     0  DD   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�����A   max       ?�e+��a     0  Et   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��Q�   max       =��     0  F�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?0��
=q   max       @FZ�G�{     �  G�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �ə����    max       @v|(�\     �  S�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @"         max       @P            �  _�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @͗        max       @� �         0  `,   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         ?i   max         ?i     0  a\   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��s�g�   max       ?�����     �  b�            &      4            ,               	                  
      
      1      &            &      "   	   �            
      +            N            $   	                  "                           (   	      %   >               
      N�GOQ�Nrk�O�2,N���Pz*BOA�N{/�N��PzA�ON�zNh�O�=�Ox��OLmOlHN	tM��8O`�4N�T�N�TBO��[N�t9N���P�R:OęO�Q�O�U\NI �Ny�O��#O!��PTIN�OP5$N��O� -N��N��UO�ON+�BM�ׂN��Pa_�ORPO7��PVO�?N=�N�O^��N�]@N�O
Nc�P�Oo��OR�hO͟Nc�ZN�rO�IQN�nOm�Oi�mN"V�N���OD.Ov�*Ng�rM�i6OF�FNDʀOx>O��xN���=�w<T��<o:�o��o�ě���`B�o�o�o�t��#�
�#�
�49X�49X�T���T���T���e`B�e`B�u�u�u��C���C���t����㼬1��1��j��j��j�ě����ͼ��ͼ�����/��/��h�����o�o�t��t��t��t���P��P��w��w��w�#�
�#�
�''''',1�,1�49X�L�ͽL�ͽY��Y��ixսq����+��+��C���C���{�� Ž� Ž�E���������������������$)6BFWYYZZOB64*)"��������������������
#/2;?A>5/#
� ������������������/;BB3
������#/<HU\aiaUH</#	���� oz}���������ztoooooo`t�������������tka]`���������������������������������������������
#
������,5?BEIKKB5)	_kt�������������~th_���
 ���������������

�����������������������������������
#/<@FIB</#���;<AHOUaafaaUH<;<><;;����������������������������������������),221,)
`gmt{����tgg````````	#Ib{�����{b0
 		��*6C\o||vh\O*��������
 ��������������������������
�����������))6BO[YOB60)))))))))^ht��������������oh^�������������������������	�����������),/+)# ������)6=?=5&������FHLTWXXWTSJHCCBEFFFFx���������������zsyx������������������������������������������������������������!*6BOhs{vqjb[OB<&!!��������}��������������������X[gght~�������tg\[XX����������������������������������������*08<IKOSUTOI<0,)%$%*����������������������������������������:;HKNKH;/2::::::::::fgkqt����vtgffffffff��������������������t���������ywtttttttt��������������������OUakida\USUQOOOOOOOOZ\t��������������whZ������
"
������%?IUbchjjgb\UIF<80$%Ubn{�����������nbRIU#)6BKGB6) ##########-0<INPLIE<830,------������"������� #*05640&#"#06803:70#jntz�����������zulij�������������������������	������#)58BFNPSOB5)&#,/8<@@AD;-#KUanz}ztnaUSKKKKKKKK

������������������������������������������/5=BNY[^[YWNKB>5-*//gjmrt��������|tg]]_g����������<�1�/�#������#�/�8�<�C�H�J�H�F�<�<�F�:�-�(�(�*�-�F�S�_�l�x�}�y�m�l�_�^�S�F�A�<�<�A�M�P�Z�f�m�f�`�]�Z�M�A�A�A�A�A�A�� �����
���$�0�=�I�N�W�W�J�=�0�$���N�L�B�>�B�N�R�[�^�g�s�t�g�[�N�N�N�N�N�N�g�5�����
���5�N�g©�g¿½����������������������������������¿�h�h�h�tāĈčĚĞĦĩĦĚčā�t�h�h�h�h���������������������������������������Ҿ��v�m�u�s�Z�O�X������վ�����𾾾����y�q�k�c�m�p�y�������������������������y������������������������������������������������������������������������������׿m�a�c�m�q�y�~�����������������������y�m��������������(�4�9�?�4�*�(�!�����������4�A�E�M�W�Z�Z�O�A�(����ùøùÿ������������ùùùùùùùùùù�"�!�����"�+�-�%�"�"�"�"�"�"�"�"�"�"������ �&�)�6�B�O�\�V�R�O�B�6�)���������������������	�	����	����������5�/�(�'�(�(�/�5�A�N�P�Y�R�N�D�A�5�5�5�5��׾ʾ����Ⱦ׾����	��!�%�#��	�����s�h�f�Z�N�M�Z�f�s�����������������s�s������������	���	�	���������������������t�K�8�+�3�g�������������������������;�.���������"�.�;�C�F�G�@�>�@�=�;��	����׾������ɾ���"�0�;�D�I�I�E�.���ƴƺ�������������������!������������������������������������������������/�/�%�#�(�/�<�?�A�A�<�0�/�/�/�/�/�/�/�/�<�7�9�?�L�F�1�,�2�<�U�a�n�z�}�f�m�a�H�<�ݿڿѿĿ������Ŀѿݿ�����	����������������������*�-�5�7�4�+�*��������ŠŝŕŔœŔśŠŭŹ����������źŹŭŠŠ����ܻڻ������@�M�Y�f�w�o�f�Y�M�4���s�l�g�\�g�s���������������������s�s�s�sĚĎġĦĭĿ����������������������ĳĦĚ�h�_�[�O�E�O�[�b�h�t�wĀāĊā�t�h�h�h�h�Z�T�S�Z�g�s�������������s�g�Z�Z�Z�Z�Z�Z�[�Z�[�a�h�tāčĚĜĦĬĦĢĚčā�t�h�[���s�k�l�q����������������������������������������
��
����������������������񻞻����������������������������������������������������������������������������������y�p�����ɺ���F�R�Z�F�:�(�����ɺ��������x�l�`�l�x���������»ǻʻƻû�����������������(�4�A�H�O�I�A�4�(�����
�������������
�#�I�U�^�^�b�l�j�U�I���������(�A�Z�g�j�q�k�g�N�I�5�(������������������������������������������������������������������������������������j�c�]�_�Y�V�?�I�b�lǈǔǞǥǣǡǔǈ�{�j�6�3�6�<�B�B�O�U�Z�[�O�B�6�6�6�6�6�6�6�6ƎƈƎƎƚơƧƭƳƵƳƭƧƚƎƎƎƎƎƎ�ʾ¾��ʾ׾�������׾ʾʾʾʾʾʾʾʺ3�'��%�)�/�.�2�L�r�~�������r�Y�E�@�-�3�������������������������ȽнֽٽֽнĽ��������������������Ľ˽нڽ�ݽʽĽ���������������ܻؼ��)�@�E�L�W�[�O�4���������������������������_�Y�U�X�_�l�x���������x�t�l�_�_�_�_�_�_���������ʼּݼ��!�.�;�<�.�!�����ּ��û����������ûлܻ����ܻлȻûûûü��������'�4�@�M�O�Q�M�M�@�4�'���Ϲù��������������ùϹܹ����������ܹ�FFFFF$F1F2F8F1F$FFFFFFFFFF�нŽĽ������������Ľнݽ����ݽսн�����������������������!�!��������E�E�E�E�E�E�E�E�E�E�E�E�E�E�FE�E�E�E�E�E�E�E�E�E�E�E�E�FFF E�E�E�E�E�E�E�E�E�ÓÒÓØàãìùûùìàÓÓÓÓÓÓÓÓ�������������������������������������������������������������������������n�g�a�]�\�a�d�n�t�zÇÒÓÙÓÍÇ�z�n�n�2�#��
��
��#�/�H�a�n�r�n�i�f�a�U�H�2��������!�.�:�;�:�9�.�%�!���� N B P  P 6 w � I 8 C [ B = U ? J U a * D & > @ O k c F _ i { U 6  " ~ Y Q L R D 3 r a P W 2 M X 5 c d X > a ` 4 6 . P % k 2 V I ; T ) S r j +  ' @ ;    �  �  �  �  �  �  �  �  �  �  �  y  �  �  �  /  3    �    ~    �  �  �  6  A  �  �    o  �      �    �  �  O  Z  >  <  �  (  �  �  �  I  L  P  :  �  �  �  �  �  �  �  �  �     �  Y  �  2  �  �  )  �  @  �  P    E  �<�����t�%   ��P�49X�ixռ�t��D���u�P�`��9X�D���C��+���
������t���o��P����ě��'ě����ͽ�+�0 Žixս�w��`B��h�u���m�h�C��"���h������w�49X����\)�\)�,1��xս}󶽉7L�}󶽓t��@��,1��C��D���e`B�49X������7L�u�e`B�e`B�8Q콗�P�m�h�y�#��vɽy�#��%�ě��J���P�����Q콸Q�\�����B�eBB�ZB�B4=BW�B��B-�B�CB  �B�B~bB̑BƂB`WB�mB}*B!I�B�:B}GBd�Bx�B�B	�@B'#zB0�{B%jB(�B��B"�B �_B*"|Be�BnB��A�
�B�{B2rBrFB��B��B
��B @�B	��B�BuB&<�BqB	A�ۥB	�yB�]B
�/B|B�7B ��B#Q�B&��B(��B�RB&QLB-{�B%VB%O�B��Bt�B��B2B��BL�BL�B<BF~B��B
)sBʦB�B3�B��B��B?�B��B�SBA�B��BG�B�:B��B�rB��B�BB�BU_B!EBįBx�B^�B��B�QB	��B'��B07B��B;B=jB �B �BB)�_BAbB�B�AA�[YB��B@B�,B�7BB�B
�NB IFB	�uBC#B��B&@
B�B��A���B	�BH&B
��B��B
eB!?�B#AB'?~B(��B�B&�QB-�	B%@B%AXB�lB��BBB8�B�TB��B?�B�B<�Ba�B
FxB½A�f@�ߍA=?B	��A�r�A���A��A���A�EAL�eAqA�@uA�f�Ao��A4Z�A7 AοA�j}A�-�A�r"A�DUAWxACR�AY�kA��A`�7AZ̕Bs�BۯA¦�AŏA|�OA��A���@�>EA���A���Aۂ�A�O9AݓA���A�<�@���A���@>�3@���A6)�A�w�A���B]�A�u�B�yAش�BduAS��?�BlA$<�A#��@�˅?#�@�%�AF-@�-[@�(�>���C��MA(#�A�[�C�^DC��iA��sA�2�A'A�YCAóA�A��%@�dA=B	�VA���A�yMA�u�Aހ%AϕQAM��Ar��A�GA�|[Ap�A5&^A8X�A΄�A�jA�A���A�rhAV'JAD(�AZƯA��NA^tA\��B�aB�A���A�[A�(A��cA�x�@Έ�A�k�A��eAڈA��,A��A���A��@�EA�e@C9�@�~�A7A��	A��2BB�A�|�B�LA�yB�.AQ�9?۷A#">A#�^@�9?/��@���A�@��K@�m�>��wC��A)
BAӀaC�`C���A˂�A�[~AC�AȃYA�y�Ah          	   '      4            ,               
                  
      
   	   2      '            &   	   #   
   �                  ,            O             $   	         	         #                     	      )   	      &   >   	            
                        3            5                                    !         9   #   %            '      %      +      #            '            9         )                        /         '         +                                                            #            /                                             5   !                           !      #            !            /         )                        #                  +                                          N�1
O=��N&��O N���PiOA�N{/�N��P(�ON�zNh�OV��N���O^6OV�N	tM��8O�3N�T�N�TBOgCN�t9N���Pm�@O�9bO���O{hNI �Ny�N��O!��O&��N�OO��_N��O� -N���N��UO�O��N+�BM�ׂN��P9AWORPO-uoP	^TO�?N=�N�N숝N�]@N�O
Nc�O��OORq)OR�hOQˣNCH�N�rO�IQNZ��Om�OX�;N"V�N���O@�O=�`Ng�rM�i6OF�FNDʀOx>OUT�N���  �  )  �  �  U  C  k  [  �  C  R  �            �  �  !  C  =  1  �  x      J  �  Y    d     X  N  Y  �  B  �  �  �  ?  �  |  �  �  C  9  y  )  �  m  /  �      �  �    i  �  {  y  ]  O  R  G  �  }  �  �  C  �  �  U  �=��<49X;�`B�49X��o�ě���`B�o�o����t��#�
�u���
�T���e`B�T���T����t��e`B�u��9X�u��C����
��1��/��9X��1��j��㼼j�0 ż��ͽ�hs������/��`B��h����w�o�o�t��<j�t���P��w��P��w��w�@��#�
�#�
�'@��0 Ž'8Q�0 Ž,1�49X�P�`�L�ͽaG��Y��ixս�+���㽇+��C���C���{�� Ž�Q콶E���������������������')6BDMOVXXYVOB5+# '��������������������	
#$/49:6/*#
		�����������������#.881&
�������#/<HU\aiaUH</#	���� oz}���������ztooooook���������������tifk���������������������������������������������

�������)57?BBB=5)t���������������vttt�������������������

����������������������������������
#/<<BC</# 
�;<AHOUaafaaUH<;<><;;����������������������������������������),221,)
`gmt{����tgg````````#Ibn{������{O/
�*6C\hvwum\O6*�����
����������������������������
�����������))6BO[YOB60)))))))))�������������������������������������������������������������),/+)# ������#+,0/+ �����FHLTWXXWTSJHCCBEFFFFx���������������zsyx������������������������������������������������������������(26BO[ioqon][OB;+&$(��������}��������������������X[gght~�������tg\[XX����������������������������������������+0<IJORTSNI<:0-)%$&+����������������������������������������:;HKNKH;/2::::::::::fgkqt����vtgffffffff��������������������t���������ywtttttttt��������������������OUakida\USUQOOOOOOOO��������������������������

��������%?IUbchjjgb\UIF<80$%bn{���������{nba\[Xb$)6BGFB6)!$$$$$$$$$$-0<INPLIE<830,------������"�������"#%02420#""""""""#06803:70#kouz�����������zwmjk�������������������������	������")5@BGMNEB5/)#/<>>?<3/&#KUanz}ztnaUSKKKKKKKK

������������������������������������������/5=BNY[^[YWNKB>5-*//jlpt����������tqgedj����������<�6�/�#������#�/�6�<�C�H�I�H�A�<�<�-�+�)�,�-�:�<�F�S�_�l�x�|�w�l�_�S�F�:�-�A�>�>�A�M�Y�Z�\�^�Z�Z�M�A�A�A�A�A�A�A�A�$����
����$�0�=�E�I�N�L�I�=�3�0�$�N�L�B�>�B�N�R�[�^�g�s�t�g�[�N�N�N�N�N�N������ �(�5�N�[�t�{�t�g�[�B�¿½����������������������������������¿�h�h�h�tāĈčĚĞĦĩĦĚčā�t�h�h�h�h���������������������������������������Ҿ��|�{������m�b�r�����ʾ�����񾾾����y�q�k�c�m�p�y�������������������������y������������������������������������������������������������������������������׿y�s�s�x�y�����������������������y�y�y�y�����������(�/�4�;�4�1�(�$������
���������(�A�C�M�Q�Y�M�L�A�(��ùøùÿ������������ùùùùùùùùùù�"�!�����"�+�-�%�"�"�"�"�"�"�"�"�"�"������#�)�)�6�B�S�R�O�O�I�B�6�2�)��������������������	�	����	����������5�/�(�'�(�(�/�5�A�N�P�Y�R�N�D�A�5�5�5�5�����׾˾Ǿо׾����	������	����s�h�f�Z�N�M�Z�f�s�����������������s�s������������	���	�	���������������������w�P�>�8�9�?�V�g���������������������.�����������"�.�4�;�B�C�@�=�<�=�;�.�����ҾǾʾ׾���	�"�.�:�A�A�8�.��	����ƹƼ�����������������
��������������������������������������������������/�/�%�#�(�/�<�?�A�A�<�0�/�/�/�/�/�/�/�/�H�E�=�<�:�<�H�M�U�a�d�k�l�f�a�U�H�H�H�H�ݿڿѿĿ������Ŀѿݿ�����	�������������������������������������ŠŝŕŔœŔśŠŭŹ����������źŹŭŠŠ�4������	���'�4�@�M�Y�d�i�h�`�Y�M�4�s�l�g�\�g�s���������������������s�s�s�sĚĎġĦĭĿ����������������������ĳĦĚ�h�g�[�O�G�O�[�g�h�t�u�~āāā�t�h�h�h�h�Z�T�S�Z�g�s�������������s�g�Z�Z�Z�Z�Z�Z�[�Z�[�a�h�tāčĚĜĦĬĦĢĚčā�t�h�[���s�p�r�y�����������������������������������������
��
����������������������񻞻������������������������������������������������������������������������������ɺ������z�������ɺ����2�5�0�"�����ɻ������x�l�`�l�x���������»ǻʻƻû����������������(�4�A�G�N�M�H�A�4�(����#��
�������������
�#�R�Z�\�_�h�f�U�I�#��������(�A�Z�g�j�q�k�g�N�I�5�(������������������������������������������������������������������������������������{�x�o�k�f�l�o�{ǈǊǔǙǠǛǔǈ�{�{�{�{�6�3�6�<�B�B�O�U�Z�[�O�B�6�6�6�6�6�6�6�6ƎƈƎƎƚơƧƭƳƵƳƭƧƚƎƎƎƎƎƎ�ʾ¾��ʾ׾�������׾ʾʾʾʾʾʾʾʺ3�-�-�5�6�;�L�e�r�����������~�r�Y�L�@�3�����������������������Žнս׽ӽнĽ����������������������Ľ˽нڽ�ݽʽĽ������������'�4�@�C�M�P�R�Q�B�@�4�'������������ ���������������_�Y�U�X�_�l�x���������x�t�l�_�_�_�_�_�_���������ʼּݼ��!�.�;�<�.�!�����ּ����������ûлܻ��ܻлû������������������������'�4�@�M�O�Q�M�M�@�4�'���Ϲù����������������ùܹ����������ܹ�FFFFF$F1F2F8F1F$FFFFFFFFFF�нŽĽ������������Ľнݽ����ݽսн�������������������������������E�E�E�E�E�E�E�E�E�E�E�E�E�FE�E�E�E�E�E�E�E�E�E�E�E�E�E�FFF E�E�E�E�E�E�E�E�E�ÓÒÓØàãìùûùìàÓÓÓÓÓÓÓÓ�������������������������������������������������������������������������n�g�a�]�\�a�d�n�t�zÇÒÓÙÓÍÇ�z�n�n�/�#������#�/�<�H�a�n�k�b�a�X�U�H�/��������!�.�:�;�:�9�.�%�!���� H @ B  P ( w � I D C [ @ ( K > J U ` * D   > @ A e _ G _ i / U 3   ~ Y Q L R C 3 r a @ W 1 O X 5 c 7 X > a 0 & 6 ' J % k 6 V F ; T " V r j +  ' ? ;    �  X  Q  �  �  �  �  �  �  �  �  �    7  �  /  3  d  �    �    �    �  f     �  �  �  o  l    �  �    �  �  O  �  >  <  �  I  �  s  �  I  L  P     �  �  �  �  �  �  �  o  �     r  Y  �  2  �  /  �  �  @  �  P    �  �  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  �  �  �  �    u  j  a  W  K  <  .    	  �  �  �  �  k  S  !  )  )  $    	  �  �  �  �  q  G    �  �  >  �  �  �    �  �  �  �  �  �  �    c  F  '    �  �  �  y  R  *      �    1  V  x  �  �  �  �  �  m  @    �  �  1  �    4  U  ^  U  L  C  8  &    �  �  �  �  t  P  >  +          �  �  �  �  �  
        >  )  �  �  X  �  �  M  �    P  =  o  k  g  _  N  8      �  �  �  x  V  8    �  �  �    )  @  [  T  N  H  B  ;  3  +  $      %  /  8  A  K  V  `  k  v  �  �  �  �  �  �  �  �  �  �  �  �  |  r  f  X  I  A  ;  6  �    .  ;  A  C  @  1        �  �  �  B  �  q  �  �  �  R  M  H  B  ?  ;  6  .  $      �  �  �  �  �  �  y  �  �  �  �  �  �  �  �  �  u  j  ^  U  P  J  D  >  9  3  -  (  "  �  �              �  �  �  �  �  l  9  �  �  �  B    �    6  N  c  q  x  }  }  x  k  V  6    �  s    �    �                  �  �  �  �  �  �  �  ^  <     �   �             �  �  �  �  t  K  O  R  3    �  �  �  u  K    x  q  j  c  Z  R  J  ?  1  #    �  �  �  �  Y  #  �  �  �  �  �  �  �  �  �  �  �  y  p  g  ^  V  O  I  C  =  7  1  �  �  �  �  �  �  �  �  }  �  |  m  ]  D     �  t  �  �   j  !                �  �  �  �  �  �  �  �  �  �  �  �  C  >  9  0  %      �  �  �  �  �  �  �  i  E    �  �  �  �    !  1  ;  =  7  -      �  �  �  r  =    �  �    �  1  %        �  �  �  �  �  �  �  �  �  p  X  ;    �  �  �  v  d  Q  =  )    �  �  �  �  �  �  p  W  +  �  �  �  m  @  x  s  j  [  J  3    �  �  m  !  �  �  O    �  '  �   �  �  �       �  �  �  �  �  �  f  D    �  �  �  :  �  �  A  �  �              �  �  �  �  {  ;  �  �  8  �  @  �  E  J  H  @  4  &    �  �  �  �  z  V  2    �  �    |   �  �  �  �  �  �  w  e  S  @  '    �  �  �  �  h  E      �   �  Y  V  S  Q  M  I  D  ?  8  1  *  "        �  �  �  �  �  �  +  K  j  r  y  �            �  �  �  c    �  H  >  d  \  S  D  1      �  �  �  �  }  d  g  j  c  X  J  6  "  �  �  �  �  �  �  �  �            �  �  �  q    |  o  X  W  U  K  A  .      �  �  �  �  �  �  l  U  D  8  Z  {  G  �  �  �    /  I  H    �  N  �  X  
�  
L  	}  �  `  m  .  Y  Q  I  A  9  1  )  #                 /  >  M  \  j  �  �  �  �  �  �  �  �  �  �  �  �  �  t  `  P  >  *  *  1  >  @  <  %    �  �  �  �  �  ]    �  �  R    �  �  4   �  �  �  �  �  �  }  k  \  M  ?  +    �  �  �  �  |  `  �  _  �  �  �  �  |  Y  7    �  �  �  �  �  �  �  �  z  p  f  \  |  �  �  �  �  �  �  �  �  �  c  B    �  �  Y  �  g  �  �  ?  ?  ?  ?  ?  ?  ?  <  9  5  2  .  *  #      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  |  q  j  l  n  p  r  t  u  |  r  g  \  T  M  E  >  8  1  .  /  /  1  6  <  D  Z  q  �  j  �  �  �  �  h  8  �  �  g    �  k  �  �    |  �  �  f  �  �  �  �  �  ~  w  �  �  �  c  /  �  �  Y    �  ;  �  �  <  B  9  (       �  �  �  |  O    �  �    �    x  �   �  6  9  8  1       �  �  �  �  �  �  e  E    �  �  D    �  y  o  h  b  X  P  N  P  K  A  *  �  �  �  I    �  o  !  �  )         �  �  �  �  �  �  n  S  7    �  �  �  �  g  A  �  �  �  �  �  �  �  �  �  w  k  _  S  F  8  )      �  �  v  �  	  T  `  l  i  X  ;    �  �  g    �  [  �  n  �  �  /  '        �  �  �  �  �  �  �  n  X  A  (    �  �  �  �  �  b  1  �  �  �  Y     �  �  o  0  �  �  h  .  
  �  �    
      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  x  �  �      �  �  �  �  �  �  }  X    �  P  �  Z  �  �  �  �  �  �  �  �  �  �  t  K     �  �  �  a    �  e  �  .  �  �  �  �  �  �  n  [  F  .    �  �  �  �  a  =    �  �  �  �  �  �        �  �  �  �  �  �  �  �  �  �  V     �  h  i  h  e  ]  N  >  -      �  �  �  �  X  ,  �  �  �  h  �  �  �  �  �  �  �  �  �  {  o  d  X  J  8  &       �   �  {  h  N  0    �  �  �  c  5  �  �  �  7  �  �  �  {  �   �  s  u  x  u  o  i  b  [  S  K  B  8  -        �  �  �  �  ]  R  G  :  *      �  �  �  ~  Q    #  %    �  �  �  i  I  O  J  <  %    �  �  �  S    �  �  C  �  K  �  �  �  �  R  N  J  H  G  F  F  E  E  F  G  F  E  A  >  -    �  S  �  G  A  ;  4  /  *  %        �  �  �  �  �  �  �  �  �  r  H  �  �  �  �  �  r  L    �  �  [    �  �  7  �  �  &  �  P  c  r  |  w  e  9  �  �  X  �  h  
�  
U  	�  	8  �  �    �  �  �  �  �  �  t  R  0    �  �  �  ^  +  �  �    �  S   �  �  �    p  `  Q  A  2  "       �  �  �  �  �  �  �  �  �  C  3  &    
  �  �  �  �  �  m  =    �  �  S    �  }  �  �  �  �  �  �  �  �  �  �  n  [  H  2       �  �  �  y  T  �  �  v  e  T  ?  (    �  �  �  �  ~  c  P  <  (          F  Q  U  Q  G  8  !    �  �  �  ]  &  �  �    �  �  �  �  �  �  �  �  �  �  �  �  �  s  c  Q  ?  .      �  �  �