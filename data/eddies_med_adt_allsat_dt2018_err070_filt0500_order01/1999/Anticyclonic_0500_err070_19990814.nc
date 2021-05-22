CDF       
      obs    >   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?��
=p��      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�G   max       P�      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ���   max       =      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?0��
=q   max       @E������     	�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�\(�    max       @v�z�G�     	�  *D   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @#         max       @P�           |  3�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @��           �  4p   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��o   max       >�~�      �  5h   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B,Z<      �  6`   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��   max       B,?�      �  7X   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >�Vt   max       C���      �  8P   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?u~   max       C��      �  9H   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max         6      �  :@   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          C      �  ;8   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          1      �  <0   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�G   max       P<�Y      �  =(   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�|����?   max       ?�b��}Vm      �  >    speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �t�   max       >%�T      �  ?   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?0��
=q   max       @E���R     	�  @   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�\(�    max       @v�z�G�     	�  I�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @!         max       @P�           |  Sp   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @�@          �  S�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         F�   max         F�      �  T�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��_o�   max       ?�\(�\     `  U�               G   6   	      %      .            =                    6   8                  "   L   	      1   +   	   .            -                                    %      9         �               OL(�M�GN�{ Nc�1P]yvO��LN���M��BO�7hO��Pj�NR�NR��NV�>PJ�Ot�|N�6�O���OU�POwN" �P��aO�F M���N<#Nr�N���N���O�"�P�N�DdN�,�P�O��N���O�5O[z	O[�)O7�O��P�Nt�GN���N��NS�NM%O6��Nf� O�VkOm�O�AO��xN��OO��OC�NN��
O��sOsx�O.�NŰ�NY��NGxƽ�㼴9X�o��`B<o<o<o<o<t�<D��<e`B<u<u<u<�o<�o<�t�<���<�1<�9X<�9X<ě�<�`B<�h<��<��<��=o=o=+=+=\)=�P=�w=#�
=49X=8Q�=<j=@�=@�=P�`=P�`=P�`=P�`=P�`=P�`=aG�=e`B=e`B=e`B=q��=u=y�#=}�=}�=�O�=�hs=���=���=���=�S�=#/7EHNQSHD</#!ttz������ttttttttttt�����������������������������������YZ^g������������tf^Y��
#/;BEE@</#
��������� ���������


 #/<HU[hfaUUH</# ��������������������������)7<9.�������

�����FHLUYagmaUKHFFFFFFFFhfknz~��znhhhhhhhhhh�������+/43,
����#/<FMPPLH</����������������������������������������526BGL[hpq{tqqkh[OB5���
#/1684/"
����������������������-*/Bgt����������gN7-��������
+-*#
�����������������������8@BOP[c[YOCB88888888c_dhrt~����uthcccccc������

��������������������������������������������������5S[[PD<3)�����������������������%),))*6BOUWWOOB62,)%���$0?BNJC)��������������������������}���������������}}}}an~���������znjijiaavz{�������������|zv`ZY[]amz��������zma`����������������������������������������)5J[tuo[NB5)
)**)%





y|����������������y���������������������������������������������������������������������

 ���#')5BMLB5)##########Q\u��������������j[Q����� !����	
')6=<:766)	5/2@T_m������zmaTH;53.)*69BEMIFB>6333333������)@NNB)��POV[ht������xwthfe[P>>DHHU_ahnqz{znaUIH>������

���������������������������������		����
")/56985)��������������������HHD<3/.//7<>EHHHHHHHEEE$E.E0E*EEEED�D�D�D�D�D�D�D�D�E�����ûɻȻû����������������������������x���z�}�x�u�l�_�S�S�R�S�Y�_�l�o�x�x�x�x�.�;�A�<�;�.�*�"�����"�*�.�.�.�.�.�.�"�H�T�^�`�\�P�F�;�"����������������	�"�m���������������y�m�`�T�N�D�<�7�@�G�`�m���������ûŻʻû������������������������Z�f�s�s�s�r�f�Z�U�R�Z�Z�Z�Z�Z�Z�Z�Z�Z�Z���������������������������������m�z���������������������z�m�`�Z�_�b�f�m�#�/�<�h�|ÃÁ�w�i�a�U�H�<�/������#�;�H�L�T�Y�Y�T�S�H�>�;�8�/�.�/�7�;�;�;�;�� �)�-�)�#���	�	������������������ �������������������������������B�[�g�s�x�t�n�m�g�a�N�)�����������)�B�����������������������������������������������!�&�#�!��
�����������"�&�8�,�/�+�"��	����ؾӾԾپ���	��"�
��#�0�<�H�V�]�U�H�<�5�/�#���	���
�5�A�B�A�A�N�[�\�R�N�A�5�(���$�(�-�4�5�Z�Z�f�g�s�y�s�f�Z�V�M�M�M�Y�Z�Z�Z�Z�Z�Z����6�L�a�a�[�O�E�N�B�*��������������������ʾ���������㾥����������������²¿��������¿²­²²²²²²²²²²²��'�(�'�'����������������'�3�@�C�L�O�L�@�9�3�1�'�&��'�'�'�'�'�'�M�Y�f�o�n�Y�M�F�@�4�0�'���$�'�4�@�K�M�(�4�A�H�G�G�A�4�(��������(�(�(�(���н����'�!�����ݽ�����������������������'�	����ƚ�u�\�C����*�hƁƳ��������������������������y�����������������������	�������ܻܻӻػܻ���������������������������������������������"�;�T�a�m�w�}�}�z�m�a�T�H�;�)�"������������������������������������������������������ںʺ������������ֺ����������������������������������ŇŔŠŹ����������ſŹŭŠŔŉŇŁŁņŇÇÓàèì÷ùþùôìàÓÇ�~�{�~ÄÇÇ�l�y�����������������y�`�S�N�?�G�S�`�i�l�������пѿÿ��������y�a�G�=�=�I�^�m�����������
�������ݿ׿ݿ������꿫���Ŀǿѿݿ�����ݿѿ˿Ŀ������������������#�$����� ����������������F1F=FJFVFXF]FVFJFDF=F1F,F1F1F1F1F1F1F1F1�ùϹܹ������ܹϹιù��ùùùùùùù��#�/�<�U�a�b�j�n�n�d�a�U�H�<�4�/�#���#����!�#�����������������(�A�M�R�V�U�J�A�4�(����������G�T�`�u�m�`�U�G�<�.�"���	��
��"�.�G�~�������������������������~�~�r�m�r�t�~čĚĦĳ������������ĳĦĚČĀ�}�}āĆč�����!�$�!��������������������������������������������r�f�^�A�@�E�S�r��f�r���������������r�e�Y�@�5�7�@�M�]�f�(�4�A�M�N�I�A�>�4�,�(���������(D�D�D�D�D�D�D�D�D�D�D�D{DsDiDfDkDoD{D�D��g�s�������������������s�k�g�Z�K�I�O�Z�g��#�0�3�<�>�@�<�0�#��� ��������
�����������������������������������������E7ECEPEWEWEPECE7E6E0E7E7E7E7E7E7E7E7E7E7���������� �*�,�*�������� ' j d q *  @ , 5 1 A < A 0 /  ? 0 K @ l " V X P 9 � . N a E 1 ( L _ X 2 )   B > b f r ` 0 Y Q P 5 M Y : T ]  3  1 2 =  �  @  �  �  �  R  �    
  4  �  k  u  i  i  �     ;  �  T  j  %  K    w  �  s  �  �  �  �    r  �  �  �  �  �  ?  �  �  �  I  �  X  �  �  �  �    N  �  �  .  �  C  �  �  q  �  s  R�D����o��o��o=��w=y�#<�C�<49X=<j=\)=q��<�C�<��
<��
=��-=C�<�`B=C�=C�=+<�`B>�~�=���=\)=�P=49X=�P=�w=��=�/='�=e`B=� �=��=H�9=�E�=��=���=�hs=�j=���=aG�=�%=e`B=ix�=e`B=��P=q��=�{=� �=��-=ě�=��P=��=� �=��
>vȴ=� �==�G�=��>�BH�B� B}eBSTB
�UB�8B#`OB$~�B�9BgJB�B�QB�GB]BշB>lB ��B �]BBm"B�~B	9_B�B�!B�BB#��B��B"BBD[Bc�B�B�]B("B��B�B ��A��B!�ZB,Z<BO�B5�B
�6B�JB�B��B�BJ�B2BǆB�A���BiBZjB%BfB�eB��B�B��B[�B�3B?�B��B�B>�B
��B��B#?�B$R�B��BPuB��B�B��B�KB��BB�B ��B �yB=B�<B�kB	<�BA�B�nB�B�}B#�
B��B"vwBB�BpB@�B��B?YBCB��B ��A�}�B!�B,?�BN7B??B?�B��B��BɲB��B�wBM&B��B�nA��B7&B�iB?�B�ZB�ZB@;B�B��B�uB�C�^@��@�7�A`5-A�tbAj�n@�akA@~7A�uA�<�A�ΌA�4�A�(!A��5A�/lA�4@Y9AZA6A�jA��SA@x�A�jOAQc�A��?�Z�?�!@��A8V�A,��B�A��X@��/A��A�Q�@��@>�A� �A��A��A^?Aoz�A�%�AyHDA�cC���>�VtA�V:A3��A7��Ab@@��A�$V@]�@�V�@߸{A7�EC�͍A���A�,�A���C��fA���C�X
@���@��PA`�zA�AkQ@�'A@��A�zRA��@Aő�A�qpA�p`A�mA���A�L�@[ʖAZ��A�qtA�PA? A�t�AR�A��G?��?��@�e�A7�A-OVB��A�о@��=A�~�A�� @��@@#A�A�A��YA�Ak@,A���Az̏A���C��?u~A;A3�A8��Aa'�@�A�%@[�O@�+�@���A8�DC���A�~�A�A�r�C��qA���               G   6   
      &      .            >                    6   8                  "   L   	      2   ,   
   .            .                                    %      9         �                              /                  '            -                     3   !                  #   C         %         '            !   +                                       %                                       !                  !                                 !                        1         !         '               +                                                               O6#M�GN�{ Nc�1O���OhX�N���M��BN��/N�U�O��@NR�NR��NV�>O�:O:L�N�6�O���OU�POwN" �O���O�L&M���N<#NIN���N��OF�vP<�YN�DdN�qVO�}IOR�N���O�5O<Y:O[�)O7�O|OP�Nt�GN��^N��NS�NM%O6��Nf� O��;O&�%O�AO��xN��OO��OC�NN��
OX�Osx�O(�NŰ�NY��NGx�  �  k  �  U  "  	  A  �  4  {  F  B  R  <  �  O  :  �  �  �      �  &  ^  �  �  �  �  �  -  �  �    �  d  (  �    �  �  �  �    N  ,  9  &  ~  �  �  �  �  �  �  �    e  �  �  �  ��t���9X�o��`B=�P<��
<o<o<�9X<���<ě�<u<u<u=�P<���<�t�<���<�1<�9X<�9X>%�T=�w<�h<��=o<��=+=0 �=]/=+=��=49X=@�=#�
=49X=@�=<j=@�=ix�=P�`=P�`=T��=P�`=P�`=P�`=aG�=e`B=u=}�=q��=u=y�#=���=}�=�O�=���=���=���=���=�S�=#/5<CHJOQH@</#ttz������ttttttttttt�����������������������������������gegtv������������tkg� 
#/5=@@<:/#
������� ���������


##%)/<@HMTQHE<//####�������������������������,1+)��������

�����FHLUYagmaUKHFFFFFFFFhfknz~��znhhhhhhhhhh��������	 
�����#/<>GHMMIH</#����������������������������������������526BGL[hpq{tqqkh[OB5���
#/1684/"
����������������������?>@EN[gt������tg[NB?�������
!"
������������������������8@BOP[c[YOCB88888888e`ehstz~�toheeeeeeee������

�������������������������������������������������5EQSNC:5)�����������������������()06BORTUOLB65.)((((�����*9AB=5)������������������������}���������������}}}}an~���������znjijiaa�{|}���������������`ZY[]amz��������zma`����������������������������������������)5J[tuo[NB5)
)**)%





|~����������������||���������������������������������������������������������������������

 ���#')5BMLB5)##########gn���������������tjg���������	
')6=<:766)	5/2@T_m������zmaTH;53.)*69BEMIFB>6333333�����*39;92)��POV[ht������xwthfe[P>>DHHU_ahnqz{znaUIH>�������� 

������������������������������������
")/56985)��������������������HHD<3/.//7<>EHHHHHHHEEE!E,E/E*E$EEEED�D�D�D�D�D�D�EE�����ûɻȻû����������������������������x���z�}�x�u�l�_�S�S�R�S�Y�_�l�o�x�x�x�x�.�;�A�<�;�.�*�"�����"�*�.�.�.�.�.�.��"�/�F�H�L�K�E�/�"�	���������������	��`�m�y���������������y�m�`�V�T�K�D�E�Q�`���������ûŻʻû������������������������Z�f�s�s�s�r�f�Z�U�R�Z�Z�Z�Z�Z�Z�Z�Z�Z�Z����������	������������������������z�������������������z�t�s�u�z�z�z�z�z�z�<�H�U�u��{�p�b�U�H�<�9�3�(�%��&�*�/�<�;�H�L�T�Y�Y�T�S�H�>�;�8�/�.�/�7�;�;�;�;�� �)�-�)�#���	�	������������������ �������������������������������)�5�B�N�a�c�d�a�[�Q�B�)��
���������)�����������������������������������������������!�&�#�!��
�����������"�&�8�,�/�+�"��	����ؾӾԾپ���	��"�
��#�0�<�H�V�]�U�H�<�5�/�#���	���
�5�A�B�A�A�N�[�\�R�N�A�5�(���$�(�-�4�5�Z�Z�f�g�s�y�s�f�Z�V�M�M�M�Y�Z�Z�Z�Z�Z�Z����)�4�>�C�B�9�)��������������������ʾ׾��������׾�����������������²¿��������¿²­²²²²²²²²²²²��'�(�'�'����������������'�3�@�B�L�N�L�@�3�'�'�#�'�'�'�'�'�'�'�'�M�Y�f�o�n�Y�M�F�@�4�0�'���$�'�4�@�K�M�(�4�A�D�E�E�A�4�(����� �(�(�(�(�(�(�ݽ�������������ݽ׽Ľ��������Ľ����������������Ƨƍ�u�h�Y�E�U�tƁƧƳ��������������������������y���������������������
������߻ܻջۻܻ��������������������������������������������������;�H�T�a�e�m�r�w�u�n�m�a�T�H�;�;�/�)�.�;���������������������������������������������������ںʺ������������ֺ����������������������������������ŇŔŠŹ����������ſŹŭŠŔŉŇŁŁņŇÇÓàèì÷ùþùôìàÓÇ�~�{�~ÄÇÇ�y�����������������y�l�`�S�O�F�G�S�`�n�y�������пѿÿ��������y�a�G�=�=�I�^�m�����������
�������ݿ׿ݿ������꿫���Ŀſѿܿݿ��ݿѿͿĿ��������������������#�$����� ����������������F1F=FJFVFXF]FVFJFDF=F1F,F1F1F1F1F1F1F1F1�ùϹܹ������ܹϹιù��ùùùùùùù��#�/�<�U�a�b�j�n�n�d�a�U�H�<�4�/�#���#����!�#����������������(�4�F�O�S�Q�E�A�4�(����������(�G�J�T�`�\�T�N�G�;�.�"���
���"�.�;�G�~�������������������������~�~�r�m�r�t�~čĚĦĳ������������ĳĦĚČĀ�}�}āĆč�����!�$�!�������������������������������������������r�f�Y�N�I�N�]�s��f�r���������������r�e�Y�@�5�7�@�M�]�f�(�4�A�M�N�I�A�>�4�,�(���������(D�D�D�D�D�D�D�D�D�D�D�D�D�D�D{DsDqDtD{D��g�s�������������������s�k�g�Z�K�I�O�Z�g��#�0�2�<�>�?�<�5�0�#��
�����
������������������������������������������E7ECEPEWEWEPECE7E6E0E7E7E7E7E7E7E7E7E7E7���������� �*�,�*�������� ' j d q '  @ ,  : 0 < A 0 $  ? 0 K @ l  P X P : � ' H F E !  - _ X - )    B > ] f r ` 0 Y 6 H 5 M Y 6 T ]  3  1 2 =  }  @  �  �  �  �  �      �  �  k  u  i  �  �     ;  �  T  j  �  K    w  h  s  �  �  Z  �  �  �  �  �  �  �  �  ?  �  �  �    �  X  �  �  �    m  N  �  �  Q  �  C  �  �  P  �  s  R  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  �  �  �  �  �  �  o  G    �  �  |  <  �  �  V    �  S  �  k    �  �  �  �  �  �  �  �    
        !  %  !      �  �  �  �  �  �  �  �  �  �  �  �  �  �    u  l  c  Z  Q  U  M  E  >  6  .  &  $  &  '  (  *  +  +  *  (  &  %  #  !    ^  �  �  �  �      "    �  �  Y    �  G  �  M  i  �  �  �  	  	  	  	  	  �  �  �  K    �  3  �  1  �  �      A  >  <  7  3      �  �  �  �  �  t  T  -    �  �  �    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  r  g  4  g  �  �  �    2  3  /  ,        �  �  +  �  $  w  �  2  �  �  �  �    2  L  b  n  v  z  {  t  d  D    �  &  �   �  $  &    A  E  @  .  !      �  �  �  �  ]    �  �    �  B  >  :  6  2  .  *  &  "       &  ,  3  9  @  F  M  S  Y  R  U  X  [  ]  Y  U  R  O  O  O  N  C  0    
  �  �  �  �  <  ;  :  9  8  5  3  0  +  "        �  �  �  �  �    _  �  �  5  o  �  �  �  �  �  �  h    �  =  �  :  �    I   �  -  @  J  N  N  M  H  A  5  $    �  �  �  �  g  '  �  j   �  :  3  -  !      �  �  �  �  �  �  �  �  �  v  R  +    �  �  �  �  �  �  �  �  �  �  �  {  ^  =    �  �  �  e  (   �  �  �  �  �  �  �  ~  U  I  Q  p  p  e  U  ?  +    �  �  �  �  �  �  �  �  z  n  c  Z  P  E  5  $    �  �  �  �  	  z      �  �  �  �  y  e  U  E  8  -  #    �  j    �  m    �  �  v  �  �  �  9  �      �  Z  �  �  �  �  �  S  \  �  �  �  �  �  �  �  �  �  �  �  ^  #  �  �  $  �  A  �    �  &            �  �  �  �  �  �  �  �  �  �  �  �  �  �  ^  T  J  @  0      �  �  �  �  �  �  t  ^  H  3  $      �  �  �  �  �  �  �  w  a  J  3      �  �  �  �  �  [    �  �  �  �  �  t  `  ]  a  f  `  P  @  /      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  l  U  E  7  *  7  g  �  �  �  �  �  �  �  �  �  �  n  =    �  �  M    �  d  }  {  z  y  �  �  �  q  X  4    �  �  W  �  �    �  )  -  -  ,  #      �  �  �  �  �  �  �    j  T  <  $    �  J  �  �  �  �  �  �  �  �  x  O    �  �  Z    �  a  �  �  �  �  �  �  �  �  �  �  �  �  �  u  T  $  �  �  >  �  �  �  a  �  �  �    �  �  �  �  �  H    �  O  �  `  �  (  c  �  �  �  �  �  i  W  H  6  $    �  �  �  �  �  �  h  G  %    d  ^  X  ^  E    �  �  |  F  
  �  ~  <  �  �  ;  �  �  �  �  "  (  %        �  �  �  �  �  z  K    �  �  L    	  �  �  �  �  �  �  t  V  5    �  �  �  R    �  �  0  ]  }      �  �  �  �  u  M    �  �  ~  J    �  �  y  ;  �  l    �  �  �  �  �  �  �  q  H  %    �  �  �  Z  �  b  �  1  �  �  �  �  �  �  o  O  )    �  �  w  6  �  �  )  �  w    �  �  �  �  �  �  �  �  �  �  �  x  c  N  8  "  
  �  �  �  �  �  �  �  �  �  �  s  Q  -    �  �  �  d  D  :  �  �  �      �  �  �  �  �  �  �  �  �  �  �  �  |  p  m  k  j  i  N  :  &          �  �  �  �  �  o  N  ,  
  �  �  �  �  ,  %          �  �  �  �  �  �  �  �  �  �  r  <    �  9  &    �  �  �  X    �  �  d  "  �  �  @  �  �  �  "  U  &               �  �  �  �  �  �  �  �  �  �  �  �  �  L  `  {  }  u  g  Q  3      �  �  �  ^    �  ^  �  �  h  �  �  �  �  �  �  �  �  j  E    �  �  n  *  �  |  �  _  �  �  �  �  �  �  �  x  a  I  0    �  �  �  �  ~  ^  =    �  �  �  �  �  �  c  =    �  �  V    �  c    �  >  �    �  �  �  �  �  }  e  F    �  �  �  �  �  �  �    i  R  ;  #     l  �  �  �  �  �  �  �  ^  ,  �  �  d  �  q    �  �  9  �  �  �  �  �  �  �  �    u  p  f  Q  +  �  o  �  �    }  �  �  �  �  �  �  �  �  x  h  V  B  .           =  l  �  �  �  j  �  �      �  �  e  �  )  c  �  �    :  �  
[  �  e  ^  V  O  H  B  5  #    �  �  �  �  �  �  h  L  /     �  �  �  �  �  �  �  �  �  �  w  T  -    �  �  �  F  �  �  �  �  �  �  �  �  �  s  [  @  !  �  �  �  i  -  �  �  �  K    �  �  v  N    �  �  w  O  2      �  �  �  �  l  7    �  �  �  a  A    �  �  �  h  2  �  �  �  B  �  �  _    �  �