CDF       
      obs    D   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�9XbM�       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N'   max       P}ϋ       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ���   max       <�t�       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>���R   max       @FU\(�     
�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��z�G�    max       @v?\(�     
�  +|   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @+         max       @Q            �  6   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @���           6�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �%   max       <D��       7�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B0��       8�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��   max       B0�D       9�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >�%$   max       C��V       :�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >e�7   max       C��       ;�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          O       =   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          7       >   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          7       ?$   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N'   max       P}ϋ       @4   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��X�e   max       ?�2a|�Q       AD   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ���   max       <�o       BT   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>���R   max       @FU\(�     
�  Cd   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���Q�     max       @v?\(�     
�  N   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @(         max       @Q            �  X�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @���           Y,   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         F#   max         F#       Z<   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�@N���U   max       ?�2a|�Q     @  [L   7      
            O                        
   -                     	               	   1      	      )      
                                                          
   $                     1   &   	         )   P�N)ןN̒�N>F�N�z�N�REP8��N@��O�5/Nk�N�i�Oҝ%N<�MO8OvnP}ϋN��N�?YN�/�Oox~O&E�N���NyK.O�?[O��N�/�N3C�N���O��vN��+N̗qN8o6O�	�O�	~N��CO���O}<QN>�}N'N�f.NJy�N�7Oi��P��O&6O��mN��O"��On��Os��O��N{�LN1W�N���O���O�N\�O0{Nd��O�M@NХO���O�G�OC�N��bOYfO�N�d�<�t�<�o<u<e`B<o<o��o�o��o��o�ě��#�
�49X�D���T���u��o��o��t���t���1��j��j�ě��ě��ě�������/��/��`B��`B��h���o�o�o�+�+�+�+�+�\)�\)�#�
�,1�,1�0 Ž0 Ž8Q�D���H�9�L�ͽT���T���aG��ixսixսixսixս}󶽃o�����+��7L��C������������rz�������������mgdgr #%0<?<<30*#        �� $)+)������� ��          36:BEOSZY[_[TOMMFB63ABOU[chohd[ROB?<<>AABN[������[B5)!#/<?@=<2/##!!!!!!!!���
#/;?:;4/#
����������������������������������������������������������������!#+0<AB<:0&#!!!!!!!!��������������������$)+6=BGOOPOFB62)'&$$#0En�������nU<0 ������������������������������������������������������������36BFQP[`nkc[OD=63233����
#$#"
�����)6=96)NO[htvtqh[OBNNNNNNNN/349DUanz����zdUH<</�����������������������������������������������������������������������������������������������[[_dghtvz|zvth[[[[[[-01<CIUbebZUNI<0.&$-������������������������(( �������������"��������#08820$## +6BO]lrsoa\RC6*)%  �	"*./,)"	������,/;<;;83//.('*,,,,,,9;BHNLIH;99999999999^abmuz|{zwsnmha_^^^^��������������������lmz��������znmcdllll��������������������OUbn{��������{ngaXPO������������������������������������������	�����������#0<IINPQMIA<70'#!#fnnt���������zspnhbf5BN[gdd][NB5)����

����������������������������agt����tgeaaaaaaaaaa��������������������S_nz��������zna`ZQKSgv��������������tgdg����������������������

���������� ����������NX\gt�����������qRNN���	���������������%)-,.)��������������(,*������������������������� ������������������������������)5AEB</	�����IQRNJH<91248<IIIIIII�����������)�B�O�R�q�t�|�t�O�?�)������/�'�"���"�.�/�;�A�;�6�/�/�/�/�/�/�/�/�����������������������������������������g�b�b�g�t�t�g�g�g�g�g�g�g�g�g�g�a�Y�U�R�U�\�a�n�z�}ÇÐÓÔÓÇ�z�x�n�a�'�$��'�-�4�=�@�C�M�S�Y�\�`�Y�M�@�4�'�'�ѿĿ��������������ѿݿ�� �������ݿ��B�A�7�?�B�O�[�]�[�Z�O�O�B�B�B�B�B�B�B�BÇÅÓÚÕÔ×àìù��������������ìàÇ�������
��!�-�/�-�(�!���������������������)�.�2�)�������������s�f�_�b�l�x�}���������ƾȾȾʾ¾��������� �!�(�!����������H�@�<�3�0�0�<�H�U�a�n�p�u�s�r�n�i�a�U�H�;�0�/�"�/�:�H�S�T�a�d�m�p�s�m�d�a�T�H�;�������m�Q�A�.��,�Z�������������������������������������������������������������Z�Y�U�Z�Z�g�m�s���������s�g�Z�Z�Z�Z�Z�Z�h�d�`�h�tāčęđčā�t�h�h�h�h�h�h�h�h�;�1�+�.�;�G�R�`�y�������������u�m�`�G�;�ʾ����������������Ⱦʾ׾�������۾׾ʹϹϹϹѹչܹ�������ܹϹϹϹϹϹϹϹϹù¹����¹ùĹϹڹٹֹϹùùùùùùù������������v�n�r�|���������������������Ż��ûлٻۻ׻׻ܻ�����$�"�	���ܻл����������������������û̻û���������������������������������������������������������������*�*�2�4�*��������:�!��#�-�:�F�S�_�l�x�����������x�_�F�:�����������������������ĽǽνĽĽ��������������!�+�-�5�:�;�:�9�6�/�-�!���n�i�k�n�{ŇŔŠŔŉŇ�{�n�n�n�n�n�n�n�n�����������������
�#�'�,�;�F�F�>�/��
���ѿ����������Ŀѿݿ�������������Ѻɺɺ����ɺֺ�������ֺɺɺɺɺɺɾʾ������������ʾ׾�����	��
�	���׾����������������������	��"�.�1�/�"��	��������������������������������������������
��������!���������������������ƾ�����������������������������O�J�O�X�\�d�h�u�y�u�h�\�O�O�O�O�O�O�O�O��������������������������������������ؼX�M�4�7�B�M�S�Y�f�r��������������r�f�X�лŻ��������ܼ�'�@�f�j�g�^�M�4�'������������źž�������������&��������[�O�;�>�C�O�T�V�\�uƎƪƶƹƳƧƎƁ�h�[���������������������������������������������� ����(�4�<�A�C�A�=�4�,�(����ù��������������ùϹܹ�� �����ܹϹ�ŠŗŔŁ�u�q�|ņŔŠŭůűŵŸżżŹŭŠ�Ľý����������Ľнݽ������ݽнĽļ����������������üʼӼʼż������������������������	���
�	���������������������	����������	�	��"�.�8�2�.�&�"���	�	�Ŀ����������������ѿ����������ѿ��0�#��������0�<�I�b�r�t�j�`�S�I�0�������������������������������������k�h�l�p���������ûлܻ�лû��������x�k���������������#��������¿¦¦²¿����������������¿�������������ɼ¼��������������������������u�n�r�|������������������������������ĿĦĚčā�k�[�T�L�R�[�h�tāĐĢķĿ��Ŀ��޹�����'�3�@�E�L�X�W�L�@�'����轷�������Ľʽнӽݽ�ݽݽнĽ������������$�#���$�0�=�I�V�b�c�b�`�X�V�I�=�0�$�$������øóó÷ù��������������� ��������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D� P = Q 7 m 4 R 3 : c , 8 7 0 B 3 8 7 4 X 6 : W 1 X 1 -  O O t d O F $ 9 b � ^ Y D / * Z p G 8  9 J , q c H < ? B � q 4 � / V | > / Z :  �  J  '  O  �    �  c  �  �  �  �  Y  �  /    �  �  �    b  �  �  �  N  �  E  �  �  �  3  �  %  �  �  E  ^  �    �  c  �  �  �  �  �  �  W  �    0  �  ~    �  ;  ~    �  �  �  �  �  K  �  W  �  Ž�P<D��;ě�<#�
��o�49X������`B�+�o�T�����T���'�9X�u��j��/��h���49X���o�8Q�8Q��w���\)�����\)��P�t���hs�}�'}�L�ͽ�P�\)��P���H�9��+��7L�ixս�7L�<j��C�����}�ixս]/�ixսy�#��^5����y�#��\)��o���T��hs��l���񪽙�������^5��F�%B �tB%�bB��B�ABs�B�B}�BO�BDB+ޙB��B d�B%�EB!�B��B'��B�vB�zB�aB��B�7B&$Bd�B=�BϿB �BCdB,$B��B��B&�IB ��BlaB�`B%_�B0��A���A��A�LA��B>lA�w:B �CB(�RB��B0BB��B&/B��B�xB#�|B+�QB	�B��B��Bl5B��B#�B#�:B	�BzBi�B'OB!��BJBe�B�B��B @�B%�8B@QBH�BA�B�B��B�fB��B+�iB��B!;dB& �B ʒBƼB(�!B��BFNB��B<�B��B�tB?aBB�B�B =�BAaB@B�B�B&B�B%jB�HB�B%\�B0�DA��A�zA�DA��B>(A�"B �uB(�
B�8B�0B�SB&?�BԫB�'B#�.B,=�B	�BB�B�B��B��B$F�B#��B	��B?�BB�B��B"R�B?�B8WB�`B�AեjA�&hA�m�A��WA�)@���A{��A�"XA�a�@_��A��AIR�A
�AŉXA��A���A���A�)$A��Ai{�AP��>�!�>�%$A�]v@�Mg@���AI��A�,�@�R�A#|d@k�KA��kA�v�A}��@;�=AS�A�c(A�A�A��B�B�eA���@ݴR@�IA���B1�A�hUA5wJ>�d�A��A)_t@�_�AY�"A]t8A|FA�y�A�_�@��@�:A��@���A�ӝA�դ?���A'�;B
��A��>C��VA�q�A��0A�q{A��=Aƅ�@�VA}�A�{4Á*@b��AԑZAJ��A��A�OeA�JA��A��A�~�Aܘ�Am �AP�>�=�>�tA�vC@���@��AI�A��w@�oA"8�@l�A�eA�ZA{(@;-ASGA��A�]0A��B��B��A��@���@��A��"B��A�mA6��>e�7A�3�A*�G@�)-AZ�VA]fBAz��A��A�[D@��@�)A���@��A��,A�u�?b�~A)?B
�Aχ�C��   8      
            O                        
   .                     
               	   1      
      )      
                                                 	         
   %                     1   &   	         )      '                  )               #            7                        %               !                                             -      !                              %                     #                  #                  %               #            7                                                                                    )                                    %                                    O��N)ןN�e�N>F�NB�N���P~�N@��Oz�Nk�N�i�Oҝ%N<�MN��nOvnP}ϋN��NPX-N�/�Oox~O�N���NyK.O:��O.�mNh_N3C�N���O~oN��+N̗qN8o6O�	�O�	~N��CO���O}<QN>�}N'N�f.NJy�N�*OY3yPM,O&6O���N�O��O6�TOs��O��N{�LN1W�N���O6�#O�N\�O0{Nd��O�M@NХO�^MO��OC�N��bOYfO�N�d�  �  i  q  �  o  I  
c  p  �  �  P  �  g  �    �  �  f  �  �  *  �  x  �  y    �      N  �  �  [  �  �    
  �  R  �  U      E  �  v  Z  D  :  A  i  �  S  �    �  a  ;  r  �  �  	  �    x  �  
  �<o<�o<e`B<e`B;ě�;D���e`B�o�t���o�ě��#�
�49X��9X�T���u��o���㼓t���t���9X��j��j����h����������`B�<j��`B��`B��h���o�o�o�+�+�+�+�+�t��t��',1�<j�49X�<j�D���D���H�9�L�ͽT���T����+�ixսixսixսixս}󶽃o��O߽�C���7L��C��������T���lqz�����������zqmiil #%0<?<<30*#        ��#"��������� ��          BBBOQXUOIB:6BBBBBBBBABOY[b`[OB??AAAAAAAABNg�������[B5)!#/<?@=<2/##!!!!!!!!���
#/4:65, �����������������������������������������������������������������!#+0<AB<:0&#!!!!!!!!��������������������$)+6=BGOOPOFB62)'&$$#0En�������nU<0 ������������������������������������������������������������36BFQP[`nkc[OD=63233���
"###"
�����)6=96)NO[htvtqh[OBNNNNNNNNEKVXanpy~zneaXULKE�����������������������������������������������������������������������������������������������[[_dghtvz|zvth[[[[[[-01<CIUbebZUNI<0.&$-������������������������(( �������������"��������#08820$## +6BO]lrsoa\RC6*)%  �	"*./,)"	������,/;<;;83//.('*,,,,,,9;BHNLIH;99999999999^abmuz|{zwsnmha_^^^^��������������������emnz������zpmdeeeeee��������������������PUbn{��������{nh]SPP�����������������������������������������������������!#.0<EILOOKI=;0)$# !npsz~����������zwson5BN[gdd][NB5)����

����������������������������agt����tgeaaaaaaaaaa��������������������T]anz������znda[VTgv��������������tgdg����������������������

���������� ����������NX\gt�����������qRNN���	���������������#'()(����������&*)����������������������������� �������������������������������):A:.)�����IQRNJH<91248<IIIIIII���������������)�B�O�b�c�O�A�6�.�����/�'�"���"�.�/�;�A�;�6�/�/�/�/�/�/�/�/�����������������������������������������g�b�b�g�t�t�g�g�g�g�g�g�g�g�g�g�U�U�U�^�a�n�zÃ�z�u�n�a�U�U�U�U�U�U�U�U�4�0�+�4�6�@�M�P�Y�X�M�@�4�4�4�4�4�4�4�4�Ŀ������������Ŀѿݿ������	����ݿѿ��B�A�7�?�B�O�[�]�[�Z�O�O�B�B�B�B�B�B�B�BàÛÞàÚØÜàìù��������������ùìà�������
��!�-�/�-�(�!���������������������)�.�2�)�������������s�f�_�b�l�x�}���������ƾȾȾʾ¾��������� �!�(�!����������H�G�=�<�9�<�H�H�U�a�k�h�j�a�a�U�H�H�H�H�;�0�/�"�/�:�H�S�T�a�d�m�p�s�m�d�a�T�H�;�������m�Q�A�.��,�Z�������������������������������������������������������������g�`�Z�Y�Z�g�s���������s�g�g�g�g�g�g�g�g�h�d�`�h�tāčęđčā�t�h�h�h�h�h�h�h�h�;�1�+�.�;�G�R�`�y�������������u�m�`�G�;�������������������ľʾ׾�����׾ʾ��ϹϹϹѹչܹ�������ܹϹϹϹϹϹϹϹϹù¹����¹ùĹϹڹٹֹϹùùùùùùù����������~��������������������������������߻޻�ܻܻ���������������點���������������»����������������������������������������������������������������� �����(�*�0�,�*��������F�@�:�:�C�F�S�_�l�x�����������x�l�_�S�F�����������������������ĽǽνĽĽ��������������!�+�-�5�:�;�:�9�6�/�-�!���n�i�k�n�{ŇŔŠŔŉŇ�{�n�n�n�n�n�n�n�n�����������������
�#�'�,�;�F�F�>�/��
���ѿ����������Ŀѿݿ�������������Ѻɺɺ����ɺֺ�������ֺɺɺɺɺɺɾʾ������������ʾ׾�����	��
�	���׾����������������������	��"�.�1�/�"��	��������������������������������������������
��������!���������������������ƾ�����������������������������O�J�O�X�\�d�h�u�y�u�h�\�O�O�O�O�O�O�O�O���������������������������������������Y�M�7�8�@�C�M�V�Y�b�r������������r�f�Y�лǻ��������ܼ�'�@�Y�e�e�]�K�4����ܻ���������źž�������������&��������h�c�Q�J�N�O�\�hƁƎƣƱƵƳƧƚƎƁ�u�h���������������������������������������������������(�4�9�A�A�;�4�(�%���Ϲù��������������ùϹܹ����������ܹ�ŠŗŔŁ�u�q�|ņŔŠŭůűŵŸżżŹŭŠ�Ľý����������Ľнݽ������ݽнĽļ����������������üʼӼʼż������������������������	���
�	���������������������	����������	�	��"�.�8�2�.�&�"���	�	�ѿĿ����������Ŀѿݿ�������������ݿ��0�#��������0�<�I�b�r�t�j�`�S�I�0�������������������������������������k�h�l�p���������ûлܻ�лû��������x�k���������������#��������¿¦¦²¿����������������¿�������������ɼ¼��������������������������y�q�t��������������������������������m�[�V�O�T�\�h�tāďĚġĶĺĳĦĚčā�m��޹�����'�3�@�E�L�X�W�L�@�'����轷�������Ľʽнӽݽ�ݽݽнĽ������������$�#���$�0�=�I�V�b�c�b�`�X�V�I�=�0�$�$��������ùôôù������������������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D� D = H 7 P , N 3 7 c , 8 7 2 B 3 8 6 4 X 4 : W 5 / - -  < O t d O F $ 9 b � ^ Y D  ) W p B D  ? J , q c H  ? B � q 4 � - G | > / X :  �  J  �  O  j  �  �  c    �  �  �  Y  �  /    �  ^  �    P  �  �  �  ~  k  E  �  _  �  3  �  %  �  �  E  ^  �    �  c  �  �  �  �  I  ;    �    0  �  ~    z  ;  ~    �  �  �  y  i  K  �  W  A  �  F#  F#  F#  F#  F#  F#  F#  F#  F#  F#  F#  F#  F#  F#  F#  F#  F#  F#  F#  F#  F#  F#  F#  F#  F#  F#  F#  F#  F#  F#  F#  F#  F#  F#  F#  F#  F#  F#  F#  F#  F#  F#  F#  F#  F#  F#  F#  F#  F#  F#  F#  F#  F#  F#  F#  F#  F#  F#  F#  F#  F#  F#  F#  F#  F#  F#  F#  F#    T  w  �  �  v  x  k  I  '  �  �  C    �  �  $  �  �  X  i  h  g  f  e  d  c  b  a  _  ^  ]  \  [  Y  X  V  T  R  P  g  l  p  f  [  H  2    �  �  �  �  �  �  S    �  [    �  �  �  �  �  �  {  s  k  c  [  U  R  P  M  J  H  G  F  D  C  !  0  G  i  q  r  o  g  ]  O  ?  +    �  �  �  G  	  �  �  �    #  1  @  I  H  I  G  ;  "    �  �  ]  !  �  �  k    
  
=  
Y  
c  
_  
O  
2  
  	�  	y  	  �  j  �  a  �  �    -  (  p  `  P  ?  /      �  �  �  �  �  �  �  �  p  �  �  �  0  �  �  �  �  �  �  �  �  �  �  t  E    �  �  \  $  *       �  �  �  �  �  �  z  q  g  ]  Q  D  6  (    	   �   �   �   �  P  G  =  4  +  !    �  �  �  �  �  �  �  j  T  >  6  3  0  �  �  �  �  �  �  �  g  D    �  �  �  W  *  �  �  �  T  Q  g  c  _  [  X  T  P  L  H  E  B  ?  =  ;  8  6  4  1  /  ,  V  |  �  �  �  �  �  �  �  �  w  Q  J     �  �  ^    �  ;        �  �  �  �  �  �  �  �  �  Y  3    �  �  �  �  t  �  �  �  �  �  �  l  P  -    �  �  j  #  �  z  #  �  d   �  �  �  �  �  �  |  i  T  @  -            *  5  0  $    ?  G  O  W  _  e  f  f  c  _  X  P  H  L  S  e  �  �  2  �  �  �  �  �  �  �  �  �  �  �  �  t  g  `  Y  Q  F  8  (    �  �  �  �  �  �  |  v  t  [  @  #    �  �  �  �  �  �  �  (  )  !      �  �  �  �  �  g  ;    �  �  I    �  �  t  �  �  q  Z  D  -       �  �  �  �  �  j  Q  6    
  �  �  x  x  w  u  r  n  j  e  a  \  W  R  M  K  J  J  I  H  H  G  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  L     �  m    7  Q  V  Q  ^  u  j  X  @  &    �  �  �  o  -  �      �  �                �  �  �  �  �  H  �  �    y  �  H  �  �  �  �  �  �    n  ^  M  ;  (      �  �  �  �  �  d  	        
    �  �  �  �  �  �  q  P  /    �  �  �  ~  �  �  �                  �  �  �  ^  �  $  8  /    N  I  D  =  2  '      �  �  �  �  �  �  y  b  J  !  �  �  �  �  �  �  �  q  [  D  -    �  �  �  �  u  J  4    �  �  �  �  �  �  �  �  �  o  \  D  *    �  �  �  �  t    �  #  [  ;    �  �  �  Z  0  �  �  }  1  �  �  k    �  >  z  �  �  �  �  �  �  �  r  X  5    �  �  v  4  �  �  m    �  %  �  �  �  �  v  k  _  S  G  :  ,        �  �  �  �  �  �              �  �  �  �  �  �  �  [  0  �  �  ,  �   �  
  �  �  �  �  �  m  K  )    �  �  �  u  @    �  �  W    �  �  �  �  |  i  Q  9  !  	  �  �  �  �  �  k  L  -     �  R  M  H  C  ?  :  5  0  +  &        �  �  �  �  �  �  �  �  �  �    |  u  e  U  E  5  $    �  �  �  �  �  �  �  k  U  J  ?  3  (    	  �  �  �  �  �  �  �  u  `  J  2      �        �  �  �  �  �  y  R  &  �  �  j    �  d    �        �  �  �  �  �  �  �  �  �  �  �  �  @  �  `  �  C  B  A  2      �  �  �  w  Y  F  -    �  �  C  �  �  �  �  �  �  }  ^  8    �  �  �  e  0  �  �  �  ^  <       (  ,  5  N  n  v  t  m  _  N  :  "    �  �  �  Y  %  �  �  �  �  W  X  X  Y  Y  Y  Z  V  P  J  D  =  7  1  )  "          3  <  B  C  6  "    �  �  �  �  �  N  
  �  ]  �  �    �    "  3  9  5  .  &    	  �  �  �  �  _  %  �  �  m  >  �  A  :  0  %    	  �  �  �  �  �  i  E  !  �  �  �  �  D  �  i  `  W  N  E  ;  4  ,      �  �  �  �  [  B  -    	   �  �  �  �  �  �  �  �  �  �      	              !  #  S  ?  +      �  �  �  �  �  {  g  F    �  �  �  z  Q  )  �  �  �  u  b  P  =  (    �  �  �  �  �  �  �  n  Y  P  G  "  w  �  �  �          �  �  �  �  I     �  4  �  =  �  �  �  �  �  �  m  O    �  �  ~  c  [  ;  #    �  �  k  D  a  W  N  E  ;  2  '        �  �  �  �  �  �  �  |  b  H  ;  7  ,      �  �  �  �  �  �  r  Z  >  #    �  ~    �  r  a  P  =  '      �  �  �  �  �  �  k  B    �  �  a     �  �  �  �  �  �  �  �  g  A    �  �  �  �  �  n  :  �  �  �  �  �  �  �  v  `  J  3       �  �  �  �  �  2  ;  /  $  �  	  	  	
  �  �  �  �  ~  <  �  {    �  %  �    �  �  �  ~  �  �  S    �  �  i  $  �  �  B  �  �    �  G  �  U  �      �  �  �  �  �  �  �  �  �  �  �  i  P  5      �  �  x  t  q  m  i  e  a  ]  X  Q  J  C  .    �  �  �  �  �  �  �  �  �  z  _  A  !  �  �  �  ~  U  2  C  7  #    �  1  �  
i  
  
o  
S  
-  	�  	�  	a  	  �  C  �  l    �    T  {  a  3  �  �  9  �  �  n  3  �  �  w  4  �  �  r  @    �  �  �  �